using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    /// <summary>
    /// Patient repository implementation.
    /// Contains patient-specific queries.
    /// </summary>
    public class PatientRepository : GenericRepository<Patient>, IPatientRepository
    {
        public PatientRepository(MedicalAssistantDbContext context) : base(context)
        {
        }

        /// <inheritdoc/>
        public async Task<Patient?> GetByEmailAsync(string email)
        {
            return await _dbSet.FirstOrDefaultAsync(p => p.Email == email);
        }

        /// <inheritdoc/>
        public async Task<Patient?> GetByPhoneNumberAsync(string phoneNumber)
        {
            return await _dbSet.FirstOrDefaultAsync(p => p.PhoneNumber == phoneNumber);
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Patient>> GetActivePatients()
        {
            return await _dbSet.Where(p => p.IsActive).ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Patient>> GetInactivePatients()
        {
            return await _dbSet.Where(p => !p.IsActive).ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Patient>> SearchByNameAsync(string name)
        {
            return await _dbSet
                .Where(p => p.FullName.Contains(name))
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<Patient>> GetByBloodTypeAsync(string bloodType)
        {
            return await _dbSet
                .Where(p => p.BloodType == bloodType)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<bool> EmailExistsAsync(string email)
        {
            return await _dbSet.AnyAsync(p => p.Email == email);
        }

        /// <inheritdoc/>
        public async Task<bool> PhoneNumberExistsAsync(string phoneNumber)
        {
            return await _dbSet.AnyAsync(p => p.PhoneNumber == phoneNumber);
        }

        /// <inheritdoc/>
        public async Task<(IEnumerable<Patient> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize)
        {
            var totalCount = await _dbSet.CountAsync();
            
            var items = await _dbSet
                .OrderByDescending(p => p.CreatedAt)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            return (items, totalCount);
        }
    }
}
