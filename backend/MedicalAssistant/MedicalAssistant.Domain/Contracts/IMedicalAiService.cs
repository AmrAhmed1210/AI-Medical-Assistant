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
}