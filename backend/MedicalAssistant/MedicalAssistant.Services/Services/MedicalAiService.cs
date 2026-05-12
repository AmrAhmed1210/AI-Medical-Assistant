using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using Microsoft.Extensions.Logging;
using System.Net.Http.Json;

namespace MedicalAssistant.Services.Services;

file record AskRequest(string Text);

public class MedicalAiService : IMedicalAiService
{
    private readonly HttpClient _http;
    private readonly ILogger<MedicalAiService> _log;

    public MedicalAiService(
        HttpClient http,
        ILogger<MedicalAiService> log)
    {
        _http = http;
        _log = log;
    }

    public async Task<string> AskAsync(
        string question,
        CancellationToken ct = default)
    {
        var result = await AskDetailedAsync(question, ct);

        return result?.GeminiReply
               ?? "Service unavailable. Please try again later.";
    }

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
                return await response.Content.ReadFromJsonAsync<AIResponseDTO>(
                    cancellationToken: ct);
            }

            _log.LogWarning(
                "Python API returned {StatusCode}",
                response.StatusCode);

            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(
                ex,
                "Failed to reach Python Medical AI service.");

            return null;
        }
        catch (TaskCanceledException)
        {
            _log.LogWarning(
                "Request to Python Medical AI service timed out.");

            return null;
        }
    }
}