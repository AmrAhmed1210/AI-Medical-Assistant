using MedicalAssistant.Domain.Entities.ConsultationsModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IConsultationRepository : IGenericRepository<Consultation>
    {
        Task<IEnumerable<Consultation>> GetByPatientIdAsync(int patientId);
        Task<IEnumerable<Consultation>> GetByDoctorIdAsync(int doctorId);
        Task<Consultation?> GetByIdWithDetailsAsync(int id);
        Task<IEnumerable<Consultation>> GetPendingByPatientIdAsync(int patientId);
    }
}
