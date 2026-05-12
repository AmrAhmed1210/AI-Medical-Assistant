using MedicalAssistant.Domain.Contracts;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ChatController : ControllerBase
{
    private readonly IMedicalAiService _medical;

    public ChatController(IMedicalAiService medical)
    {
        _medical = medical;
    }

    // POST api/chat/ask
    // Body: { "question": "I have a headache and fever" }
    [HttpPost("ask")]
    public async Task<IActionResult> Ask([FromBody] ChatRequest req, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Question))
            return BadRequest(new { error = "Question cannot be empty." });

        var reply = await _medical.AskAsync(req.Question, ct);
        return Ok(new ChatResponse(reply));
    }

    // POST api/chat/ask-detailed  → returns matches + confidence too
    [HttpPost("ask-detailed")]
    public async Task<IActionResult> AskDetailed([FromBody] ChatRequest req, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Question))
            return BadRequest(new { error = "Question cannot be empty." });

        var result = await _medical.AskDetailedAsync(req.Question, ct);
        if (result is null)
            return StatusCode(503, new { error = "Medical AI service is currently unavailable." });

        return Ok(result);
    }
}

public record ChatRequest(string Question);
public record ChatResponse(string Reply);
