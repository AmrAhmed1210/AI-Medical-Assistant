using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using MedicalAssistant.Shared.DTOs.SessionDTOs;
using Microsoft.AspNetCore.Http;

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
        CancellationToken ct);
}