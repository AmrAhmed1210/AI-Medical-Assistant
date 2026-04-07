using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class AppointmentRepository(MedicalAssistantDbContext context) : GenericRepository<Appointment>(context), IAppointmentRepository
    {
        public async Task<IEnumerable<Appointment>> GetByPatientIdAsync(int patientId)
        {
            return await _dbSet.Where(a => a.PatientId == patientId && !a.IsDeleted).ToListAsync();
        }

        public async Task<IEnumerable<Appointment>> GetByDoctorIdAsync(int doctorId)
        {
            return await _dbSet.Where(a => a.DoctorId == doctorId && !a.IsDeleted).ToListAsync();
        }

        public async Task<Appointment?> GetByIdAsync(int id)
        {
            return await _dbSet.FirstOrDefaultAsync(a => a.Id == id && !a.IsDeleted);
        }

        public async Task<(IEnumerable<Appointment> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize)
        {
            var totalCount = await _dbSet.Where(a => !a.IsDeleted).CountAsync();
            var items = await _dbSet
                .Where(a => !a.IsDeleted)
                .OrderByDescending(a => a.ScheduledAt)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();
            return (items, totalCount);
        }
    }
}
