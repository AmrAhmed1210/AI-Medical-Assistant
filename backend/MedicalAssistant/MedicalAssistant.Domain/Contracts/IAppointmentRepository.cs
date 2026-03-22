using MedicalAssistant.Domain.Entities.AppointmentsModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IAppointmentRepository : IGenericRepository<Appointment>
    {
        Task<IEnumerable<Appointment>> GetByPatientIdAsync(int patientId);

        /// <summary>
        /// Returns appointments for a patient including Doctor + Specialty navigation
        /// </summary>
        Task<IEnumerable<Appointment>> GetByPatientIdWithDoctorAsync(int patientId);

        Task<IEnumerable<Appointment>> GetByDoctorIdAsync(int doctorId);

        Task<Appointment?> GetByIdAsync(int id);

        /// <summary>
        /// Returns single appointment including Doctor + Specialty navigation
        /// </summary>
        Task<Appointment?> GetByIdWithDoctorAsync(int id);

        Task<(IEnumerable<Appointment> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize);
    }
}
