using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class DoctorRepository : GenericRepository<Doctor>, IDoctorRepository
    {
        public DoctorRepository(MedicalAssistantDbContext context) : base(context)
        {
        }

        public async Task<IEnumerable<Doctor>> GetAvailableDoctorsAsync()
        {
            return await _dbSet
                .Include(d => d.Specialty)
                .Where(d => d.IsAvailable)
                .OrderByDescending(d => d.Rating)
                .ToListAsync();
        }

        public async Task<IEnumerable<Doctor>> GetUnavailableDoctorsAsync()
        {
            return await _dbSet
                .Include(d => d.Specialty)
                .Where(d => !d.IsAvailable)
                .OrderByDescending(d => d.Rating)
                .ToListAsync();
        }

        public async Task<IEnumerable<Doctor>> GetBySpecialtyAsync(int specialtyId)
        {
            return await _dbSet
                .Include(d => d.Specialty)
                .Where(d => d.SpecialtyId == specialtyId)
                .OrderByDescending(d => d.Rating)
                .ToListAsync();
        }

        public async Task<IEnumerable<Doctor>> SearchByNameAsync(string name)
        {
            return await _dbSet
                .Include(d => d.Specialty)
                .Where(d => d.Name.Contains(name))
                .OrderByDescending(d => d.Rating)
                .ToListAsync();
        }

        public async Task<IEnumerable<Doctor>> GetTopRatedDoctorsAsync(int count)
        {
            return await _dbSet
                .Include(d => d.Specialty)
                .OrderByDescending(d => d.Rating)
                .Take(count)
                .ToListAsync();
        }

        public async Task<(IEnumerable<Doctor> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize)
        {
            var totalCount = await _dbSet.CountAsync();

            var items = await _dbSet
                .Include(d => d.Specialty)
                .OrderByDescending(d => d.Rating)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            return (items, totalCount);
        }
    }
}