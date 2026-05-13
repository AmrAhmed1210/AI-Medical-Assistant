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

    [HttpPost("ask")]
    [ProducesResponseType(typeof(ChatResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (req is null || string.IsNullOrWhiteSpace(req.Question))
        {
            return BadRequest(new ErrorResponse("Question cannot be empty."));
        }

        var result = await _medical.AskDetailedAsync(
            req.Question.Trim(),
            req.History,
            ct
        );

        if (result is null)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("Medical AI service is currently unavailable."));
        }

        return Ok(new ChatResponse(result.GeminiReply));
    }

    [HttpPost("ask-detailed")]
    [ProducesResponseType(typeof(AIResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (req is null || string.IsNullOrWhiteSpace(req.Question))
        {
            return BadRequest(new ErrorResponse("Question cannot be empty."));
        }

        var result = await _medical.AskDetailedAsync(
            req.Question.Trim(),
            req.History,
            ct
        );

        if (result is null)
        {
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("Medical AI service is currently unavailable."));
        }

        return Ok(result);
    }

    [HttpPost("analyze-image")]
    [ProducesResponseType(typeof(MedicalAnalysisResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
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
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("Medical image service is currently unavailable."));
        }

        return Ok(result);
    }

    [HttpGet("health")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public IActionResult Health()
    {
        return Ok(new
        {
            status = "ok",
            service = "Medical AI API",
            uptime = DateTime.UtcNow
        });
    }
}

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