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
    private readonly IMedicalAiService _medicalAiService;

    public ChatController(IMedicalAiService medicalAiService)
    {
        _medicalAiService = medicalAiService;
    }

    [HttpPost("ask")]
    [ProducesResponseType(typeof(ChatResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest request,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var reply = await _medicalAiService.AskAsync(request.Question, ct);

        return Ok(new ChatResponse(reply));
    }

    [HttpPost("ask-detailed")]
    [ProducesResponseType(typeof(MedicalAnalysisResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest request,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(request.Question))
            return BadRequest(new ErrorResponse("Question cannot be empty."));

        var result = await _medicalAiService.AskDetailedAsync(request.Question, ct);

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
    [ProducesResponseType(typeof(MedicalAnalysisResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AnalyzeImage(
        [FromForm] AnalyzeImageRequest request,
        CancellationToken ct)
    {
        if (request.Image is null || request.Image.Length == 0)
        {
            return BadRequest(new ErrorResponse("No image file provided."));
        }

        var allowedTypes = new[]
        {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic"
        };

        if (!allowedTypes.Contains(request.Image.ContentType))
        {
            return BadRequest(
                new ErrorResponse(
                    "Unsupported file type. Please upload a JPEG, PNG, WEBP, or HEIC image."));
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