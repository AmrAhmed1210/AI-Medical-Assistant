using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace MedicalAssistant.Services.Services;

file record AskRequest(
    [property: JsonPropertyName("text")] string Text
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
        CancellationToken ct = default)
    {
        var result = await AskDetailedAsync(question, ct);

        return result?.GeminiReply ?? ServiceUnavailableMessage;
    }

    /// <inheritdoc />
    public async Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        CancellationToken ct = default)
    {
        try
        {
            var response = await _http.PostAsJsonAsync(
                "/ask",
                new AskRequest(question),
                ct);

            if (response.IsSuccessStatusCode)
            {
                var dto = await response.Content
                    .ReadFromJsonAsync<AIResponseDTO>(cancellationToken: ct);

                if (dto is null)
                {
                    _log.LogWarning(
                        "Python AI service returned an empty response body for question: {Question}",
                        question);
                }

                return dto;
            }

            if (response.StatusCode == HttpStatusCode.UnprocessableEntity)
            {
                _log.LogWarning(
                    "Validation error from Python AI service. StatusCode: {StatusCode} — Question: {Question}",
                    response.StatusCode,
                    question);
            }
            else
            {
                _log.LogWarning(
                    "Python AI service returned an unexpected status. StatusCode: {StatusCode}",
                    response.StatusCode);
            }

            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(
                ex,
                "Failed to reach Python Medical AI service. Question: {Question}",
                question);

            return null;
        }
        catch (TaskCanceledException ex) when (!ct.IsCancellationRequested)
        {
            _log.LogWarning(
                ex,
                "Request to Python Medical AI service timed out. Question: {Question}",
                question);

            return null;
        }
    }
}