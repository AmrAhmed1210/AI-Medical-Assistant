using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;
using Microsoft.AspNetCore.Http;

namespace MedicalAssistant.Domain.Contracts;

public interface IMedicalAiService
{
    Task<string> AskAsync(string question, CancellationToken ct = default);

    Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        CancellationToken ct = default);

    Task<MedicalAnalysisResponseDTO?> AnalyzeMedicalImageAsync(IFormFile file, CancellationToken ct = default);
}