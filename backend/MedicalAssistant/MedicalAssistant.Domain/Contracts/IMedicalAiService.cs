using MedicalAssistant.Shared.DTOs.AIChatBotDTOs;

namespace MedicalAssistant.Domain.Contracts;

public interface IMedicalAiService
{
    Task<string> AskAsync(string question, CancellationToken ct = default);

    Task<AIResponseDTO?> AskDetailedAsync(
        string question,
        CancellationToken ct = default);
}