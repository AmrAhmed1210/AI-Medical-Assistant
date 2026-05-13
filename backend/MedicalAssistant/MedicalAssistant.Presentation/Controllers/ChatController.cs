using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
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
    private readonly IMedicalAiService _medical;

    public ChatController(IMedicalAiService medical)
    {
        _medical = medical;
    }

    /// <summary>
    /// Ask the medical AI a question and get a plain text reply.
    /// </summary>
    /// <param name="req">The question to ask.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A simple text reply from the AI.</returns>
    [HttpPost("ask")]
    [ProducesResponseType(typeof(ChatResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var reply = await _medical.AskAsync(req.Question, ct);
        return Ok(new ChatResponse(reply));
    }

    /// <summary>
    /// Ask the medical AI a question and get a detailed response
    /// including matches, confidence scores, and database status.
    /// </summary>
    /// <param name="req">The question to ask.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A detailed AI response with matches and metadata.</returns>
    [HttpPost("ask-detailed")]
    [ProducesResponseType(typeof(AIResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var result = await _medical.AskDetailedAsync(req.Question, ct);

        if (result is null)
            return StatusCode(
                StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("Medical AI service is currently unavailable."));

        return Ok(result);
    }
}

/// <summary>Request body for chat endpoints.</summary>
public record ChatRequest(
    [property: Required]
    [property: JsonPropertyName("question")]
    string Question
);

/// <summary>Simple reply response.</summary>
public record ChatResponse(
    [property: JsonPropertyName("reply")]
    string Reply
);

/// <summary>Error response wrapper.</summary>
public record ErrorResponse(
    [property: JsonPropertyName("error")]
    string Error
);