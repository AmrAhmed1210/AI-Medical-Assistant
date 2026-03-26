namespace MedicalAssistant.Shared.DTOs.Admin;

public class ModelReloadDto
{
    public string ModelName { get; set; } = string.Empty;
    public bool IsSuccess { get; set; }
    public string Message { get; set; } = string.Empty;
}