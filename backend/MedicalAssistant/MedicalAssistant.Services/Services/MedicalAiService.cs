using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MedicalAssistant.Services.Services;

// ─────────────────────────────────────────────────────────────────
// Internal request record — matches Python /ask payload exactly.
// v13.0: history is now forwarded so Python receives full context.
// ─────────────────────────────────────────────────────────────────
file sealed record PythonAskRequest(
    [property: JsonPropertyName("question")] string Question,
    [property: JsonPropertyName("history")] List<MessageDto>? History
);

// ─────────────────────────────────────────────────────────────────
// MedicalAiService
// ─────────────────────────────────────────────────────────────────
public sealed class MedicalAiService : IMedicalAiService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private const string ServiceUnavailableEn = "Service unavailable. Please try again later.";

    private readonly HttpClient _http;
    private readonly ILogger<MedicalAiService> _log;

    public MedicalAiService(HttpClient http, ILogger<MedicalAiService> log)
    {
        _http = http;
        _log = log;
    }

    // ─────────────────────────────────────────────────────────────
    // AskAsync — simple string reply
    // ─────────────────────────────────────────────────────────────
    public async Task<string> AskAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct)
    {
        var result = await AskDetailedAsync(question, history, ct);
        return result?.Reply ?? ServiceUnavailableEn;
    }

    // ─────────────────────────────────────────────────────────────
    // AskDetailedAsync — full RAG response DTO
    // ─────────────────────────────────────────────────────────────
    public async Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(question))
        {
            _log.LogWarning("[MedicalAiService] AskDetailedAsync called with empty question.");
            return null;
        }

        try
        {
            // v13.0: history now included in the payload sent to Python
            var payload = new PythonAskRequest(question.Trim(), history);
            var response = await _http.PostAsJsonAsync("/ask", payload, JsonOptions, ct);

            if (!response.IsSuccessStatusCode)
            {
                _log.LogWarning(
                    "[MedicalAiService] /ask returned {StatusCode} for question: {Question}",
                    (int)response.StatusCode,
                    Truncate(question, 80));

                return null;
            }

            var dto = await response.Content
                .ReadFromJsonAsync<AIResponseDTO>(JsonOptions, ct);

            if (dto is null)
            {
                _log.LogWarning("[MedicalAiService] /ask returned null DTO.");
                return null;
            }

            _log.LogInformation(
                "[MedicalAiService] /ask success — model={Model} lang={Lang} " +
                "isMedical={IsMedical} foundInDb={FoundInDb} historyTurns={Turns}",
                dto.ModelUsed,
                dto.Language,
                dto.IsMedical,
                dto.FoundInDatabase,
                history?.Count ?? 0);

            return dto;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _log.LogInformation("[MedicalAiService] /ask cancelled by client.");
            return null;
        }
        catch (TaskCanceledException)
        {
            _log.LogWarning("[MedicalAiService] /ask timed out.");
            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(ex, "[MedicalAiService] /ask HTTP error: {Message}", ex.Message);
            return null;
        }
        catch (JsonException ex)
        {
            _log.LogError(ex, "[MedicalAiService] /ask JSON deserialization failed.");
            return null;
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[MedicalAiService] /ask unexpected error.");
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // AnalyzeMedicalImageAsync — multipart image upload
    // (unchanged from v12.0 — no contract change needed here)
    // ─────────────────────────────────────────────────────────────
    public async Task<MedicalAnalysisResponseDTO?> AnalyzeMedicalImageAsync(
        IFormFile file,
        string? patientContext = null,
        CancellationToken ct = default)
    {
        if (file is null || file.Length == 0)
        {
            _log.LogWarning("[MedicalAiService] AnalyzeMedicalImage called with null or empty file.");
            return null;
        }

        try
        {
            using var multipart = new MultipartFormDataContent();

            using var ms = new MemoryStream((int)file.Length);
            await file.CopyToAsync(ms, ct);
            ms.Position = 0;

            var fileContent = new StreamContent(ms);
            fileContent.Headers.ContentType = MediaTypeHeaderValue.TryParse(
                file.ContentType, out var parsed)
                ? parsed
                : new MediaTypeHeaderValue("application/octet-stream");

            multipart.Add(fileContent, "file", file.FileName ?? "upload.jpg");
            if (!string.IsNullOrWhiteSpace(patientContext))
            {
                multipart.Add(new StringContent(patientContext), "patient_context");
            }

            var response = await _http.PostAsync("/analyze-image", multipart, ct);

            var body = await response.Content.ReadAsStringAsync(ct);

            if (string.IsNullOrWhiteSpace(body))
            {
                _log.LogWarning(
                    "[MedicalAiService] /analyze-image returned empty body with status {Status}",
                    (int)response.StatusCode);
                return null;
            }

            MedicalAnalysisResponseDTO? dto;
            try
            {
                dto = JsonSerializer.Deserialize<MedicalAnalysisResponseDTO>(body, JsonOptions);
            }
            catch (JsonException ex)
            {
                _log.LogError(ex,
                    "[MedicalAiService] /analyze-image JSON parse failed. Body: {Body}",
                    Truncate(body, 200));
                return null;
            }

            if (dto is null)
            {
                _log.LogWarning("[MedicalAiService] /analyze-image returned null DTO.");
                return null;
            }

            _log.LogInformation(
                "[MedicalAiService] /analyze-image success — status={Status} model={Model}",
                dto.Status,
                dto.ModelUsed ?? "unknown");

            return dto;
        }
        catch (TaskCanceledException) when (ct.IsCancellationRequested)
        {
            _log.LogInformation("[MedicalAiService] /analyze-image cancelled by client.");
            return null;
        }
        catch (TaskCanceledException)
        {
            _log.LogWarning("[MedicalAiService] /analyze-image timed out.");
            return null;
        }
        catch (HttpRequestException ex)
        {
            _log.LogError(ex, "[MedicalAiService] /analyze-image HTTP error: {Message}", ex.Message);
            return null;
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[MedicalAiService] /analyze-image unexpected error.");
            return null;
        }
    }

    public async Task<object?> AnalyzePatientHistoryAsync(
        object historyPayload,
        CancellationToken ct = default) =>
        await PostJsonAsync<object>("/analyze-history", historyPayload, ct);

    public async Task<object?> SummarizeSurgeryAsync(
        string description,
        CancellationToken ct = default) =>
        await PostJsonAsync<object>("/summarize-surgery", new { description }, ct);

    public async Task<object?> SummarizeMedicalItemAsync(
        string type,
        string description,
        CancellationToken ct = default) =>
        await PostJsonAsync<object>("/summarize-medical-item", new { type, description }, ct);

    public async Task<object?> AnalyzeVitalsAsync(
        object payload,
        CancellationToken ct = default) =>
        await PostJsonAsync<object>("/analyze-vitals", payload, ct);

    public async Task<object?> CheckMedicationSafetyAsync(
        object payload,
        CancellationToken ct = default) =>
        await PostJsonAsync<object>("/check-medication-safety", payload, ct);

    public async Task<object?> ParseMedicalProfileAsync(
        string text,
        CancellationToken ct = default) =>
        await PostJsonAsync<object>("/parse-medical-profile", new { text }, ct);

    private async Task<T?> PostJsonAsync<T>(
        string path,
        object payload,
        CancellationToken ct) where T : class
    {
        try
        {
            var response = await _http.PostAsJsonAsync(path, payload, JsonOptions, ct);
            if (!response.IsSuccessStatusCode)
            {
                _log.LogWarning(
                    "[MedicalAiService] {Path} returned {StatusCode}",
                    path,
                    (int)response.StatusCode);
                return null;
            }

            return await response.Content.ReadFromJsonAsync<T>(JsonOptions, ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[MedicalAiService] {Path} failed", path);
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────
    private static string Truncate(string value, int maxLength) =>
        value.Length <= maxLength ? value : string.Concat(value.AsSpan(0, maxLength), "…");
}