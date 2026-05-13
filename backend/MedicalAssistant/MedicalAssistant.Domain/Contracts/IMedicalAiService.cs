using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Http;

namespace MedicalAssistant.Domain.Contracts;

using MessageDto = MedicalAssistant.Shared.DTOs.AIChatDTOs.MessageDto;

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