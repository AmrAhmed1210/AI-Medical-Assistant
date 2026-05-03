namespace MedicalAssistant.Shared.DTOs.Secretary;

public record SecretaryDto(int Id, int UserId, int DoctorId, string FullName, string Email, bool IsActive);

public record CreateSecretaryDto(string FullName, string Email, string Password);
