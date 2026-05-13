using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Services_Abstraction.Contracts;
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

    public ChatController(
        IMedicalAiService medicalAiService,
        ISessionService sessionService,
        IMessageService messageService)
    {
        _medicalAiService = medicalAiService;
        _sessionService = sessionService;
        _messageService = messageService;
    }

    private int GetUserIdFromClaims()
    {
        var claim = User.FindFirst("UserId")?.Value;
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
    [ProducesResponseType(typeof(ChatResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest request,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var userId = GetUserIdFromClaims();
        if (userId <= 0) return Unauthorized();

        int sessionId = request.SessionId ?? 0;
        if (sessionId == 0)
        {
            var session = await _sessionService.CreateSessionAsync(userId, request.Question.Length > 30 ? request.Question[..30] + "..." : request.Question, "AI");
            sessionId = session.Id;
        }

        // 1. Get History
        var history = await _messageService.GetMessagesForSessionAsync(sessionId);

        // 2. Save User Message
        await _messageService.SendMessageAsync(sessionId, userId, "user", request.Question);

        // 3. Ask AI with History
        var reply = await _medicalAiService.AskAsync(request.Question, history.ToList(), ct);

        // 4. Save AI Reply
        await _messageService.SendMessageAsync(sessionId, 0, "assistant", reply);

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

        return Ok(result);
    }
}

public sealed class AnalyzeImageRequest
{
    [Required]
    public IFormFile? Image { get; set; }
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