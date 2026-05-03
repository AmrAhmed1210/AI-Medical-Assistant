using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Shared.DTOs.Secretary;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface ISecretaryService
{
    Task<SecretaryDto> AddSecretaryAsync(int doctorUserId, CreateSecretaryDto dto);
    Task<IEnumerable<SecretaryDto>> GetSecretariesForDoctorAsync(int doctorUserId);
    Task<bool> DeleteSecretaryAsync(int doctorUserId, int secretaryId);
    Task<int?> GetDoctorIdForSecretaryAsync(int secretaryUserId);
}
