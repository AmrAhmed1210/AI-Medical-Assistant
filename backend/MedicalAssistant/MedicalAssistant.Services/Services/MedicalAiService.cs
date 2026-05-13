using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.Http.Json;

namespace MedicalAssistant.Services.Services;

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

    public async Task<string> AskAsync(string question, CancellationToken ct = default)
    {
        var result = await AskDetailedAsync(question, ct);
        return result?.GeminiReply ?? ServiceUnavailableMessage;
    }

    public async Task<AIResponseDTO?> AskDetailedAsync(string question, CancellationToken ct = default)
    {
        try
        {
            var response = await _http.PostAsJsonAsync(
                "/ask",
                new { text = question },
                ct);

            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadFromJsonAsync<AIResponseDTO>(cancellationToken: ct);
            }

            if (response.StatusCode == HttpStatusCode.UnprocessableEntity)
            {
                _log.LogWarning("Validation error from AI service. Question: {Question}", question);
            }
            else
            {
                _log.LogWarning("AI service returned status {StatusCode}", response.StatusCode);
            }

            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(ex, "AI service unreachable. Question: {Question}", question);
            return null;
        }
        catch (TaskCanceledException ex) when (!ct.IsCancellationRequested)
        {
            _log.LogWarning(ex, "AI service timeout. Question: {Question}", question);
            return null;
        }
    }
}