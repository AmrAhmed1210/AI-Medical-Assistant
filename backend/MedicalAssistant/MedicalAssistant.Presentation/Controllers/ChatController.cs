using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Text.Json.Serialization;
using MessageDto = MedicalAssistant.Shared.DTOs.AIChatDTOs.MessageDto;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class ChatController : ControllerBase
{
    private readonly IMedicalAiService _medical;

    public ChatController(IMedicalAiService medical)
    {
        _medical = medical;
    }

    // ─────────────────────────────────────────────
    // SIMPLE CHAT
    // ─────────────────────────────────────────────
    [HttpPost("ask")]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req?.Question))
        {
            return BadRequest(new ErrorResponse("Question cannot be empty."));
        }

        var result = await _medical.AskDetailedAsync(
            req.Question.Trim(),
            req.History,
            ct);

        if (result is null)
        {
            return StatusCode(503, new ErrorResponse("Service unavailable."));
        }

        return Ok(new ChatResponse(result.Reply));
    }

    // ─────────────────────────────────────────────
    // DETAILED CHAT (RAG OUTPUT)
    // ─────────────────────────────────────────────
    [HttpPost("ask-detailed")]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req?.Question))
        {
            return BadRequest(new ErrorResponse("Question cannot be empty."));
        }

        var result = await _medical.AskDetailedAsync(
            req.Question.Trim(),
            req.History,
            ct);

        if (result is null)
        {
            return StatusCode(503, new ErrorResponse("Service unavailable."));
        }

        return Ok(result);
    }

    // ─────────────────────────────────────────────
    // IMAGE ANALYSIS
    // ─────────────────────────────────────────────
    [HttpPost("analyze-image")]
    public async Task<IActionResult> AnalyzeImage(
        IFormFile file,
        CancellationToken ct)
    {
        if (file is null || file.Length == 0)
        {
            return BadRequest(new ErrorResponse("Image file is required."));
        }

        var result = await _medical.AnalyzeMedicalImageAsync(file, ct);

        if (result is null)
        {
            return StatusCode(503, new ErrorResponse("Service unavailable."));
        }

        return Ok(result);
    }

    // ─────────────────────────────────────────────
    // HEALTH CHECK
    // ─────────────────────────────────────────────
    [HttpGet("health")]
    public IActionResult Health()
    {
        return Ok(new
        {
            status = "ok",
            service = "Medical AI API"
        });
    }
}

// ─────────────────────────────────────────────
// DTOs
// ─────────────────────────────────────────────
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