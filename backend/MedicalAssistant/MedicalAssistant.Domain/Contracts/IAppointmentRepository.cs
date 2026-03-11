using MedicalAssistant.Domain.Entities.AppointmentsModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IAppointmentRepository : IGenericRepository<Appointment>
    {
        Task<IEnumerable<Appointment>> GetByPatientIdAsync(int patientId);
        Task<IEnumerable<Appointment>> GetByDoctorIdAsync(int doctorId);
        Task<Appointment?> GetByIdAsync(int id);
        Task<(IEnumerable<Appointment> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize);
    }
}
