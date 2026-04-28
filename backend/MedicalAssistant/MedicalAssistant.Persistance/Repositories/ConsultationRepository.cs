using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.ConsultationsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories;

public class ConsultationRepository : GenericRepository<Consultation>, IConsultationRepository
{
    public ConsultationRepository(MedicalAssistantDbContext context) : base(context)
    {
    }

    public async Task<IEnumerable<Consultation>> GetByPatientIdAsync(int patientId)
    {
        return await _dbSet
            .Where(c => c.PatientId == patientId)
            .Include(c => c.Doctor)
            .Include(c => c.Patient)
            .OrderByDescending(c => c.CreatedAt)
            .ToListAsync();
    }

    public async Task<IEnumerable<Consultation>> GetByDoctorIdAsync(int doctorId)
    {
        return await _dbSet
            .Where(c => c.DoctorId == doctorId)
            .Include(c => c.Doctor)
            .Include(c => c.Patient)
            .OrderByDescending(c => c.CreatedAt)
            .ToListAsync();
    }

    public async Task<Consultation?> GetByIdWithDetailsAsync(int id)
    {
        return await _dbSet
            .Include(c => c.Doctor)
            .Include(c => c.Patient)
            .FirstOrDefaultAsync(c => c.Id == id);
    }

    public async Task<IEnumerable<Consultation>> GetPendingByPatientIdAsync(int patientId)
    {
        return await _dbSet
            .Where(c => c.PatientId == patientId && c.Status == "Scheduled")
            .Include(c => c.Doctor)
            .OrderByDescending(c => c.ScheduledAt)
            .ToListAsync();
    }
}
