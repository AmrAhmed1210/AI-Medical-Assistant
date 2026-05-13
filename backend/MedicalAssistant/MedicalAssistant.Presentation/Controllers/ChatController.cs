using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
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

    [HttpPost("ask")]
    public async Task<IActionResult> Ask([FromBody] ChatRequest req, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var reply = await _medical.AskAsync(req.Question, ct);
        return Ok(new ChatResponse(reply));
    }

    [HttpPost("ask-detailed")]
    public async Task<IActionResult> AskDetailed([FromBody] ChatRequest req, CancellationToken ct)
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

public record ChatRequest(
    [property: JsonPropertyName("question")]
    string Question
);

public record ChatResponse(
    [property: JsonPropertyName("reply")]
    string Reply
);

public record ErrorResponse(
    [property: JsonPropertyName("error")]
    string Error
);