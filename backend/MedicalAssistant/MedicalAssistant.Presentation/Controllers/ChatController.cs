using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System.Text.Json.Serialization;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public sealed class ChatController : ControllerBase
{
    private readonly IMedicalAiService _medical;
    private readonly ILogger<ChatController> _log;

    public ChatController(IMedicalAiService medical, ILogger<ChatController> log)
    {
        _medical = medical;
        _log = log;
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/ask
    // Simple chat — returns a plain reply string
    // ─────────────────────────────────────────────────────────────
    [HttpPost("ask")]
    [ProducesResponseType(typeof(ChatResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (!IsValidRequest(req, out var validationError))
            return BadRequest(new ErrorResponse(validationError));

        _log.LogInformation(
            "[ChatController] ask — question='{Question}'",
            Truncate(req.Question, 80));

        var result = await _medical.AskDetailedAsync(req.Question.Trim(), req.History, ct);

        if (result is null)
        {
            _log.LogWarning("[ChatController] ask — AI service returned null.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("الخدمة غير متاحة مؤقتاً. يرجى المحاولة لاحقاً."));
        }

        return Ok(new ChatResponse(result.Reply ?? string.Empty));
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/ask-detailed
    // Full RAG response — returns the complete AIResponseDTO
    // ─────────────────────────────────────────────────────────────
    [HttpPost("ask-detailed")]
    [ProducesResponseType(typeof(AIResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (!IsValidRequest(req, out var validationError))
            return BadRequest(new ErrorResponse(validationError));

        _log.LogInformation(
            "[ChatController] ask-detailed — question='{Question}'",
            Truncate(req.Question, 80));

        var result = await _medical.AskDetailedAsync(req.Question.Trim(), req.History, ct);

        if (result is null)
        {
            _log.LogWarning("[ChatController] ask-detailed — AI service returned null.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("الخدمة غير متاحة مؤقتاً. يرجى المحاولة لاحقاً."));
        }

        return Ok(result);
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/analyze-image
    // Medical image analysis — multipart/form-data
    // ─────────────────────────────────────────────────────────────
    [HttpPost("analyze-image")]
    [ProducesResponseType(typeof(MedicalAnalysisResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AnalyzeImage(
        IFormFile? file,
        CancellationToken ct)
    {
        if (file is null || file.Length == 0)
        {
            _log.LogWarning("[ChatController] analyze-image — no file provided.");
            return BadRequest(new ErrorResponse("يرجى إرفاق ملف صورة صحيح."));
        }

        _log.LogInformation(
            "[ChatController] analyze-image — file='{FileName}' size={Size}KB type={Type}",
            file.FileName,
            file.Length / 1024,
            file.ContentType);

        var result = await _medical.AnalyzeMedicalImageAsync(file, ct);

        if (result is null)
        {
            _log.LogWarning("[ChatController] analyze-image — AI service returned null.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة لاحقاً."));
        }

        return Ok(result);
    }

    // ─────────────────────────────────────────────────────────────
    // GET api/chat/health
    // ─────────────────────────────────────────────────────────────
    [HttpGet("health")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public IActionResult Health() =>
        Ok(new { status = "ok", service = "Medical AI API" });

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────
    private static bool IsValidRequest(ChatRequest? req, out string error)
    {
        if (req is null || string.IsNullOrWhiteSpace(req.Question))
        {
            error = "السؤال مطلوب ولا يمكن أن يكون فارغاً.";
            return false;
        }

        if (req.Question.Length > 500)
        {
            error = "السؤال طويل جداً. الحد الأقصى 500 حرف.";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private static string Truncate(string value, int maxLength) =>
        value.Length <= maxLength ? value : string.Concat(value.AsSpan(0, maxLength), "…");
}

// ─────────────────────────────────────────────────────────────────
// DTOs — local to presentation layer
// ─────────────────────────────────────────────────────────────────
public sealed record ChatRequest(
    [property: JsonPropertyName("question")]
    string Question,

    [property: JsonPropertyName("history")]
    List<MessageDto>? History
);

public sealed record ChatResponse(
    [property: JsonPropertyName("reply")]
    string Reply
);

public sealed record ErrorResponse(
    [property: JsonPropertyName("error")]
    string Error
);