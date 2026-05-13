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
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("history")] List<AskHistoryItem>? History = null
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

    /// <inheritdoc />
    public async Task<string> AskAsync(
        string question,
        List<MessageDto>? history = null,
        CancellationToken ct = default)
    {
        var result = await AskDetailedAsync(question, history, ct);
        return result?.GeminiReply ?? ServiceUnavailableMessage;
    }

    /// <inheritdoc />
    public async Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        List<MessageDto>? history = null,
        CancellationToken ct = default)
    {
        try
        {
            var geminiHistory = history?
                .Select(m => new AskHistoryItem(
                    m.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? "model" : "user",
                    new List<string> { m.Role.Equals("system", StringComparison.OrdinalIgnoreCase) ? $"[SYSTEM INSTRUCTION]: {m.Content}" : m.Content }
                ))
                .ToList();

            var response = await _http.PostAsJsonAsync(
                "/ask",
                new AskRequest(question, geminiHistory),
                ct);

            if (response.IsSuccessStatusCode)
            {
                var dto = await response.Content
                    .ReadFromJsonAsync<AIResponseDTO>(cancellationToken: ct);

                if (dto is null)
                    _log.LogWarning(
                        "Python AI service returned an empty response body for question: {Question}",
                        question);

                return dto;
            }

            if (response.StatusCode == HttpStatusCode.UnprocessableEntity)
                _log.LogWarning(
                    "Validation error from Python AI service. StatusCode: {StatusCode} — Question: {Question}",
                    response.StatusCode, question);
            else
                _log.LogWarning(
                    "Python AI service returned an unexpected status. StatusCode: {StatusCode}",
                    response.StatusCode);

            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(ex,
                "Failed to reach Python Medical AI service. Question: {Question}", question);
            return null;
        }
        catch (TaskCanceledException ex) when (!ct.IsCancellationRequested)
        {
            _log.LogWarning(ex,
                "Request to Python Medical AI service timed out. Question: {Question}", question);
            return null;
        }
    }

    /// <inheritdoc />
    public async Task<MedicalAnalysisResponseDTO?> AnalyzeMedicalImageAsync(
        IFormFile file,
        CancellationToken ct = default)
    {
        try
        {
            // ── Build multipart form ────────────────────────────────────
            using var content = new MultipartFormDataContent();
            using var fileStream = file.OpenReadStream();
            using var fileContent = new StreamContent(fileStream);

            fileContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);

            content.Add(fileContent, name: "file", fileName: file.FileName);

            // ── Send to Python /analyze-image ───────────────────────────
            var response = await _http.PostAsync("/analyze-image", content, ct);

            if (response.IsSuccessStatusCode)
            {
                var dto = await response.Content
                    .ReadFromJsonAsync<MedicalAnalysisResponseDTO>(cancellationToken: ct);

                if (dto is null)
                    _log.LogWarning(
                        "Python AI service returned empty body for image analysis. File: {FileName}",
                        file.FileName);

                return dto;
            }

            _log.LogWarning(
                "Python AI service returned {StatusCode} for image analysis. File: {FileName}",
                response.StatusCode, file.FileName);

            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(ex,
                "Failed to reach Python Medical AI service for image analysis. File: {FileName}",
                file.FileName);
            return null;
        }
        catch (TaskCanceledException ex) when (!ct.IsCancellationRequested)
        {
            _log.LogWarning(ex,
                "Image analysis request timed out. File: {FileName}", file.FileName);
            return null;
        }
    }
}  