using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories;

public class DoctorRepository : GenericRepository<Doctor>, IDoctorRepository
{
    public DoctorRepository(MedicalAssistantDbContext context) : base(context) { }

    public async Task<IEnumerable<Doctor>> GetAvailableDoctorsAsync()
    {
        return await _dbSet.Include(d => d.Specialty).Where(d => d.IsAvailable).ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> GetUnavailableDoctorsAsync()
    {
        return await _dbSet.Include(d => d.Specialty).Where(d => !d.IsAvailable).ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> GetBySpecialtyAsync(int specialtyId)
    {
        return await _dbSet.Include(d => d.Specialty).Where(d => d.SpecialtyId == specialtyId).ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> SearchByNameAsync(string name)
    {
        return await _dbSet.Include(d => d.Specialty).Where(d => d.Name.Contains(name)).ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> GetTopRatedDoctorsAsync(int count)
    {
        return await _dbSet.Include(d => d.Specialty).OrderByDescending(d => d.Rating).Take(count).ToListAsync();
    }

    public async Task<(IEnumerable<Doctor> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize)
    {
        var total = await _dbSet.CountAsync();
        var items = await _dbSet.Include(d => d.Specialty).Skip((pageNumber - 1) * pageSize).Take(pageSize).ToListAsync();
        return (items, total);
    }

    public async Task<IEnumerable<DoctorAvailability>> GetAvailabilityAsync(int doctorId)
    {
        return await _context.Set<DoctorAvailability>().Where(a => a.DoctorId == doctorId).ToListAsync();
    }

    public async Task UpdateAvailabilityAsync(int doctorId, IEnumerable<DoctorAvailability> slots)
    {
        var existing = await _context.Set<DoctorAvailability>().Where(a => a.DoctorId == doctorId).ToListAsync();
        _context.Set<DoctorAvailability>().RemoveRange(existing);
        await _context.Set<DoctorAvailability>().AddRangeAsync(slots);
        await _context.SaveChangesAsync();
    }
}