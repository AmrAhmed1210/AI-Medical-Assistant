namespace MedicalAssistant.Shared.DTOs.Admin;
public class ModelReloadResponse
{
    public string AgentName { get; set; } = string.Empty;
    public bool IsSuccess { get; set; }
    public string Message { get; set; } = string.Empty;
}