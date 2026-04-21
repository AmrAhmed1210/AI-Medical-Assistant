using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class AppointmentRepository(MedicalAssistantDbContext context)
        : GenericRepository<Appointment>(context), IAppointmentRepository
    {
        public async Task<IEnumerable<Appointment>> GetByPatientIdAsync(int patientId)
        {
            return await _dbSet
                .Where(a => a.PatientId == patientId)
                .ToListAsync();
        }

        public async Task<IEnumerable<Appointment>> GetByPatientIdWithDoctorAsync(int patientId)
        {
            return await _dbSet
                .Where(a => a.PatientId == patientId)
                .Include(a => a.Patient)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.Specialty)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.User)
                .OrderByDescending(a => a.CreatedAt)
                .ToListAsync();
        }

        public async Task<IEnumerable<Appointment>> GetByDoctorIdAsync(int doctorId)
        {
            return await _dbSet
                .Where(a => a.DoctorId == doctorId)
                .Include(a => a.Patient)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.Specialty)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.User)
                .OrderByDescending(a => a.CreatedAt)
                .ToListAsync();
        }

        public new async Task<Appointment?> GetByIdAsync(int id)
        {
            return await _dbSet.FirstOrDefaultAsync(a => a.Id == id);
        }

        public async Task<Appointment?> GetByIdWithDoctorAsync(int id)
        {
            return await _dbSet
                .Include(a => a.Patient)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.Specialty)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.User)
                .FirstOrDefaultAsync(a => a.Id == id);
        }

        public async Task<(IEnumerable<Appointment> Items, int TotalCount)> GetPaginatedAsync(
            int pageNumber, int pageSize)
        {
            var totalCount = await _dbSet.CountAsync();
            var items = await _dbSet
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.Specialty)
                .Include(a => a.Doctor)
                    .ThenInclude(d => d.User)
                .OrderByDescending(a => a.CreatedAt)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();
            return (items, totalCount);
        }
    }
}
