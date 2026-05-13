using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Services_Abstraction.DTOs;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using MedicalAssistant.Shared.DTOs.SessionDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace MedicalAssistant.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class ChatController : ControllerBase
{
    private readonly IMedicalAiService _medicalAiService;
    private readonly ISessionService _sessionService;
    private readonly IMessageService _messageService;
    private readonly IPatientRecordService _recordService;

    public ChatController(
        IMedicalAiService medicalAiService,
        ISessionService sessionService,
        IMessageService messageService,
        IPatientRecordService recordService)
    {
        _medicalAiService = medicalAiService;
        _sessionService = sessionService;
        _messageService = messageService;
        _recordService = recordService;
    }

    private int GetUserIdFromClaims()
    {
        var claim = User.FindFirst("UserId")?.Value;
        return int.TryParse(claim, out var id) ? id : 0;
    }

    private int GetPatientIdFromClaims()
    {
        var claim = User.FindFirst("PatientId")?.Value;
        return int.TryParse(claim, out var id) ? id : 0;
    }

    [HttpGet("sessions")]
    public async Task<IActionResult> GetSessions()
    {
        var userId = GetUserIdFromClaims();
        if (userId <= 0) return Unauthorized();
        var sessions = await _sessionService.GetSessionsByUserIdAsync(userId);
        return Ok(sessions.Where(s => s.Type == "AI"));
    }

    [HttpGet("sessions/{sessionId:int}/messages")]
    public async Task<IActionResult> GetMessages(int sessionId)
    {
        var messages = await _messageService.GetMessagesForSessionAsync(sessionId);
        return Ok(messages);
    }

    [HttpPost("ask")]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest request,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var userId = GetUserIdFromClaims();
        var patientId = GetPatientIdFromClaims();
        if (userId <= 0) return Unauthorized();

        int sessionId = request.SessionId ?? 0;
        bool isNewSession = sessionId == 0;

        if (isNewSession)
        {
            var session = await _sessionService.CreateSessionAsync(userId, request.Question.Length > 30 ? request.Question[..30] + "..." : request.Question, "AI");
            sessionId = session.Id;
        }

        // 1. Get History (Limit to last 10)
        var allHistory = await _messageService.GetMessagesForSessionAsync(sessionId);
        var history = allHistory.TakeLast(10).ToList();

        // 2. If new session, inject Patient Context
        if (isNewSession && patientId > 0)
        {
            try {
                var medicalHistory = await _recordService.GetPatientCompleteHistoryAsync(patientId);
                var contextPrompt = "SYSTEM CONTEXT: You are chatting with a patient. Here is their medical background: " +
                                   $"Chronic Diseases: {string.Join(", ", medicalHistory.ChronicDiseases.Select(d => d.DiseaseName))}; " +
                                   $"Allergies: {string.Join(", ", medicalHistory.Allergies.Select(a => a.AllergenName))}; " +
                                   $"Current Medications: {string.Join(", ", medicalHistory.Medications.Select(m => m.MedicationName))}. " +
                                   "Use this information to provide personalized advice, but do not repeat the full list unless asked. Be professional and empathetic.";
                
                // Add as a hidden model instruction or a system message if supported. 
                // For now, we prepend it to history if possible or just as a virtual message.
                history.Insert(0, new MessageDto { Role = "system", Content = contextPrompt, Timestamp = DateTime.UtcNow });
            } catch { /* proceed without context if fetch fails */ }
        }

        // 3. Parallelize: Save User Message AND Ask AI
        var saveUserTask = _messageService.SendMessageAsync(sessionId, userId, "user", request.Question);
        var aiTask = _medicalAiService.AskAsync(request.Question, history, ct);

        await Task.WhenAll(saveUserTask, aiTask);
        var reply = await aiTask;

        // 4. Save AI Reply in background
        _ = _messageService.SendMessageAsync(sessionId, 0, "assistant", reply);

        return Ok(new ChatResponse(reply, sessionId));
    }

    [HttpPost("ask-detailed")]
    [ProducesResponseType(typeof(AIResponseDTO), StatusCodes.Status200OK)]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest request,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var userId = GetUserIdFromClaims();
        var history = request.SessionId.HasValue 
            ? await _messageService.GetMessagesForSessionAsync(request.SessionId.Value) 
            : new List<MessageDto>();

        var result = await _medicalAiService.AskDetailedAsync(request.Question, history.ToList(), ct);

        if (result is null)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("Medical AI service is currently unavailable."));
        }

        return Ok(result);
    }

    [HttpPost("analyze-image")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> AnalyzeImage(
        [FromForm] AnalyzeImageRequest request,
        CancellationToken ct)
    {
        if (request.Image is null || request.Image.Length == 0)
        {
            return BadRequest(new ErrorResponse("No image file provided."));
        }

        var result = await _medicalAiService.AnalyzeMedicalImageAsync(request.Image, ct);

        if (result is null)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("Medical image analysis service is currently unavailable."));
        }

        // If a SessionId is provided, save the analysis result as a message in the chat
        if (request.SessionId.HasValue && request.SessionId.Value > 0)
        {
            var userId = GetUserIdFromClaims();
            var analysisText = $"[تحليل الصورة الطبية]\n{result.Analysis}\n\n[Medical Details]\n{result.TechnicalDetails}";
            
            // Save User's image message (placeholder for now as text)
            await _messageService.SendMessageAsync(request.SessionId.Value, userId, "user", "[Sent an image for analysis]");
            
            // Save AI's analysis
            await _messageService.SendMessageAsync(request.SessionId.Value, 0, "assistant", analysisText);
        }

        return Ok(result);
    }
}

public sealed class AnalyzeImageRequest
{
    [Required]
    public IFormFile? Image { get; set; }
    public int? SessionId { get; set; }
}

public record ChatRequest(
    [property: JsonPropertyName("question")]
    string Question,
    [property: JsonPropertyName("sessionId")]
    int? SessionId = null
);

public record ChatResponse(
    [property: JsonPropertyName("reply")]
    string Reply,
    [property: JsonPropertyName("sessionId")]
    int SessionId
);

public record ErrorResponse(
    [property: JsonPropertyName("error")]
    string Error
);