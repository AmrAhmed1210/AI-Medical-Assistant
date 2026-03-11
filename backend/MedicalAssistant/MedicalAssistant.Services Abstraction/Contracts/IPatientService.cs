using MedicalAssistant.Shared.DTOs.PatientDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    /// <summary>
    /// Patient service contract. Defines available operations for managing patients.
    /// (API endpoints/controllers are not included in this module).
    /// </summary>
    public interface IPatientService
    {
        /// <summary>
        /// Returns all patients.
        /// </summary>
        Task<IEnumerable<PatientDto>> GetAllPatientsAsync();

        /// <summary>
        /// Returns a patient by id.
        /// </summary>
        Task<PatientDto?> GetPatientByIdAsync(int id);

        /// <summary>
        /// Returns a patient by email.
        /// </summary>
        Task<PatientDto?> GetPatientByEmailAsync(string email);

        /// <summary>
        /// Returns a patient by phone number.
        /// </summary>
        Task<PatientDto?> GetPatientByPhoneNumberAsync(string phoneNumber);

        /// <summary>
        /// Returns active patients.
        /// </summary>
        Task<IEnumerable<PatientDto>> GetActivePatientsAsync();

        /// <summary>
        /// Returns inactive patients.
        /// </summary>
        Task<IEnumerable<PatientDto>> GetInactivePatientsAsync();

        /// <summary>
        /// Searches patients by full name.
        /// </summary>
        Task<IEnumerable<PatientDto>> SearchPatientsByNameAsync(string name);

        /// <summary>
        /// Returns patients filtered by blood type.
        /// </summary>
        Task<IEnumerable<PatientDto>> GetPatientsByBloodTypeAsync(string bloodType);

        /// <summary>
        /// Returns patients with pagination.
        /// </summary>
        Task<PaginatedResultDto<PatientDto>> GetPaginatedPatientsAsync(int pageNumber, int pageSize);

        /// <summary>
        /// Creates a new patient.
        /// </summary>
        Task<PatientDto> CreatePatientAsync(CreatePatientDto createPatientDto);

        /// <summary>
        /// Updates an existing patient.
        /// </summary>
        Task<PatientDto?> UpdatePatientAsync(UpdatePatientDto updatePatientDto);

        /// <summary>
        /// Deletes a patient.
        /// </summary>
        Task<bool> DeletePatientAsync(int id);

        /// <summary>
        /// Activates a patient account.
        /// </summary>
        Task<bool> ActivatePatientAsync(int id);

        /// <summary>
        /// Deactivates a patient account.
        /// </summary>
        Task<bool> DeactivatePatientAsync(int id);

        /// <summary>
        /// Checks whether an email already exists.
        /// </summary>
        Task<bool> EmailExistsAsync(string email);

        /// <summary>
        /// Checks whether a phone number already exists.
        /// </summary>
        Task<bool> PhoneNumberExistsAsync(string phoneNumber);
    }
}
