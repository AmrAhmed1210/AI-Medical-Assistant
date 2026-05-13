using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.Services.Services;

file record AskRequest(
    [property: JsonPropertyName("question")] string Question,
    [property: JsonPropertyName("history")] List<AskHistoryItem>? History
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

    public MedicalAiService(HttpClient http, ILogger<MedicalAiService> log)
    {
        _http = http;
        _log = log;
    }

    public async Task<string> AskAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct)
    {
        var result = await AskDetailedAsync(question, history, ct);
        return result?.GeminiReply ?? ServiceUnavailableMessage;
    }

    public async Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct)
    {
        try
        {
            var geminiHistory = history?
                .Select(m => new AskHistoryItem(
                    m.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? "model" : "user",
                    new List<string>
                    {
                        m.Role.Equals("system", StringComparison.OrdinalIgnoreCase)
                            ? $"[SYSTEM INSTRUCTION]: {m.Content}"
                            : m.Content
                    }
                ))
                .ToList();

            var response = await _http.PostAsJsonAsync(
                "/ask",
                new AskRequest(question, geminiHistory),
                ct);

            if (response.IsSuccessStatusCode)
            {
                return await response.Content
                    .ReadFromJsonAsync<AIResponseDTO>(cancellationToken: ct);
            }

            _log.LogWarning(
                "AI service returned {StatusCode} for question: {Question}",
                response.StatusCode, question);

            return null;
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "AI service failure. Question: {Question}", question);
            return null;
        }
    }

    public async Task<MedicalAnalysisResponseDTO?> AnalyzeMedicalImageAsync(
        IFormFile file,
        CancellationToken ct)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var fileStream = file.OpenReadStream();
            using var fileContent = new StreamContent(fileStream);

            fileContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);

            content.Add(fileContent, "file", file.FileName);

            var response = await _http.PostAsync("/analyze-image", content, ct);

            if (response.IsSuccessStatusCode)
            {
                return await response.Content
                    .ReadFromJsonAsync<MedicalAnalysisResponseDTO>(cancellationToken: ct);
            }

            _log.LogWarning(
                "Image analysis failed with {StatusCode} for {FileName}",
                response.StatusCode, file.FileName);

            return null;
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Image analysis error for {FileName}", file.FileName);
            return null;
        }
    }
}