using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace MedicalAssistant.Services.Services;

file record AskRequest(
    [property: JsonPropertyName("question")] string Question
);

file record AskHistoryItem(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("parts")] List<string> Parts
);

public sealed class MedicalAiService : IMedicalAiService
{
    private const string ServiceUnavailableMessage =
        "Service unavailable. Please try again later.";

    private readonly HttpClient _http;
    private readonly ILogger<MedicalAiService> _log;

    public MedicalAiService(
        HttpClient http,
        ILogger<MedicalAiService> log)
    {
        _http = http;
        _log = log;
    }

    // ─────────────────────────────────────────────
    // SIMPLE ASK
    // ─────────────────────────────────────────────
    public async Task<string> AskAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct)
    {
        var result = await AskDetailedAsync(question, history, ct);
        return result?.Reply ?? ServiceUnavailableMessage;
    }

    // ─────────────────────────────────────────────
    // DETAILED ASK (RAG RESPONSE)
    // ─────────────────────────────────────────────
    public async Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct)
    {
        try
        {
            // NOTE: history is ignored for now (backend no longer supports it)
            var request = new AskRequest(question);

            var response = await _http.PostAsJsonAsync(
                "/ask",
                request,
                ct);

            if (!response.IsSuccessStatusCode)
            {
                _log.LogWarning(
                    "AI service returned {StatusCode}",
                    response.StatusCode);

                return null;
            }

            return await response.Content.ReadFromJsonAsync<AIResponseDTO>(
                cancellationToken: ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "AI service failure");
            return null;
        }
    }

    // ─────────────────────────────────────────────
    // IMAGE ANALYSIS
    // ─────────────────────────────────────────────
    public async Task<MedicalAnalysisResponseDTO?> AnalyzeMedicalImageAsync(
        IFormFile file,
        CancellationToken ct)
    {
        try
        {
            using var content = new MultipartFormDataContent();

            await using var stream = file.OpenReadStream();

            using var fileContent = new StreamContent(stream);

            fileContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);

            content.Add(fileContent, "file", file.FileName);

            var response = await _http.PostAsync(
                "/analyze-image",
                content,
                ct);

            if (!response.IsSuccessStatusCode)
            {
                _log.LogWarning(
                    "Image analysis failed with {StatusCode}",
                    response.StatusCode);

                return null;
            }

            return await response.Content
                .ReadFromJsonAsync<MedicalAnalysisResponseDTO>(
                    cancellationToken: ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Image analysis error");
            return null;
        }
    }
}