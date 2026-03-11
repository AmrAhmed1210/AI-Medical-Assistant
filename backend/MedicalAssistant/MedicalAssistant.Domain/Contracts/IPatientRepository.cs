using MedicalAssistant.Domain.Entities.PatientModule;

namespace MedicalAssistant.Domain.Contracts
{
    /// <summary>
    /// Patient repository contract.
    /// Provides patient-specific operations in addition to the generic repository operations.
    /// </summary>
    public interface IPatientRepository : IGenericRepository<Patient>
    {
        /// <summary>
        /// Returns a patient by email.
        /// </summary>
        Task<Patient?> GetByEmailAsync(string email);

        /// <summary>
        /// Returns a patient by phone number.
        /// </summary>
        Task<Patient?> GetByPhoneNumberAsync(string phoneNumber);

        /// <summary>
        /// Returns active patients.
        /// </summary>
        Task<IEnumerable<Patient>> GetActivePatients();

        /// <summary>
        /// Returns inactive patients.
        /// </summary>
        Task<IEnumerable<Patient>> GetInactivePatients();

        /// <summary>
        /// Searches patients by full name.
        /// </summary>
        Task<IEnumerable<Patient>> SearchByNameAsync(string name);

        /// <summary>
        /// Returns patients filtered by blood type.
        /// </summary>
        Task<IEnumerable<Patient>> GetByBloodTypeAsync(string bloodType);

        /// <summary>
        /// Checks whether an email already exists.
        /// </summary>
        Task<bool> EmailExistsAsync(string email);

        /// <summary>
        /// Checks whether a phone number already exists.
        /// </summary>
        Task<bool> PhoneNumberExistsAsync(string phoneNumber);

        /// <summary>
        /// Returns patients with pagination.
        /// </summary>
        Task<(IEnumerable<Patient> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize);
    }
}
