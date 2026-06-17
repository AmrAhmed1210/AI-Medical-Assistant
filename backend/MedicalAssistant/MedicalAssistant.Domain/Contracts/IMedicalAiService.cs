using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Http;
using MessageDto = MedicalAssistant.Shared.DTOs.AIChatDTOs.MessageDto;

namespace MedicalAssistant.Domain.Contracts;

public interface IMedicalAiService
{
    Task<string> AskAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct);

    Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        List<MessageDto>? history,
        CancellationToken ct);

    Task<MedicalAnalysisResponseDTO?> AnalyzeMedicalImageAsync(
        IFormFile file,
        string? patientContext = null,
        CancellationToken ct = default);

    Task<object?> AnalyzePatientHistoryAsync(
        object historyPayload,
        CancellationToken ct = default);

    Task<object?> SummarizeSurgeryAsync(
        string description,
        CancellationToken ct = default);

    Task<object?> SummarizeMedicalItemAsync(
        string type,
        string description,
        CancellationToken ct = default);

    Task<object?> AnalyzeVitalsAsync(
        object payload,
        CancellationToken ct = default);

    Task<object?> CheckMedicationSafetyAsync(
        object payload,
        CancellationToken ct = default);

    Task<object?> ParseMedicalProfileAsync(
        string text,
        CancellationToken ct = default);

    Task<object?> GeneratePreVisitSummaryAsync(
        object payload,
        CancellationToken ct = default);

    Task<object?> GetPersonalizedTipAsync(
        object payload,
        CancellationToken ct = default);
}