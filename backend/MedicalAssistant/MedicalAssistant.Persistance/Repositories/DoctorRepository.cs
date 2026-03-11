using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    /// <summary>
    /// Doctor repository implementation.
    /// Contains doctor-specific queries.
    /// </summary>
    public class DoctorRepository : GenericRepository<Doctor>, IDoctorRepository
    {
        public DoctorRepository(MedicalAssistantDbContext context) : base(context)
        {
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Doctor>> GetAvailableDoctorsAsync()
        {
            return await _dbSet
                .Where(d => d.IsAvailable)
                .Include(d => d.Specialty)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Doctor>> GetUnavailableDoctorsAsync()
        {
            return await _dbSet
                .Where(d => !d.IsAvailable)
                .Include(d => d.Specialty)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Doctor>> GetBySpecialtyAsync(int specialtyId)
        {
            return await _dbSet
                .Where(d => d.SpecialtyId == specialtyId)
                .Include(d => d.Specialty)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Doctor>> SearchByNameAsync(string name)
        {
            return await _dbSet
                .Where(d => d.Name.Contains(name))
                .Include(d => d.Specialty)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Doctor>> GetTopRatedDoctorsAsync(int count)
        {
            return await _dbSet
                .OrderByDescending(d => d.Rating)
                .Take(count)
                .Include(d => d.Specialty)
                .ToListAsync();
        }

        /// <inheritdoc/>
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