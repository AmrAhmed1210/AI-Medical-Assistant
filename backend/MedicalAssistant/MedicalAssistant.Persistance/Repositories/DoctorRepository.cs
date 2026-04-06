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
        return await _dbSet
            .Include(d => d.Specialty)
            .Include(d => d.User)
            .ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> GetBySpecialtyAsync(int specialtyId)
    {
        return await _dbSet
            .Include(d => d.Specialty)
            .Include(d => d.User)
            .Where(d => d.SpecialtyId == specialtyId)
            .ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> SearchByNameAsync(string name)
    {
        return await _dbSet
            .Include(d => d.Specialty)
            .Include(d => d.User)
            .Where(d => d.User.FullName.Contains(name))
            .ToListAsync();
    }

    public async Task<IEnumerable<Doctor>> GetTopRatedDoctorsAsync(int count)
    {
        return await _dbSet
            .Include(d => d.Specialty)
            .Include(d => d.User)
            .OrderByDescending(d => d.Reviews.Any() ? d.Reviews.Average(r => r.Rating) : 0)
            .Take(count)
            .ToListAsync();
    }

    public async Task<(IEnumerable<Doctor> Items, int TotalCount)> GetPaginatedAsync(int pageNumber, int pageSize)
    {
        var total = await _dbSet.CountAsync();
        var items = await _dbSet
            .Include(d => d.Specialty)
            .Include(d => d.User)
            .Skip((pageNumber - 1) * pageSize)
            .Take(pageSize)
            .ToListAsync();
        return (items, total);
    }

    public async Task<IEnumerable<DoctorAvailability>> GetAvailabilityAsync(int doctorId)
    {
        return await _context.Set<DoctorAvailability>()
            .Where(a => a.DoctorId == doctorId)
            .ToListAsync();
    }

    public async Task UpdateAvailabilityAsync(int doctorId, IEnumerable<DoctorAvailability> slots)
    {
        var existing = await _context.Set<DoctorAvailability>()
            .Where(a => a.DoctorId == doctorId)
            .ToListAsync();

        _context.Set<DoctorAvailability>().RemoveRange(existing);

        foreach (var slot in slots) slot.DoctorId = doctorId;

        await _context.Set<DoctorAvailability>().AddRangeAsync(slots);
    }

    public async Task<Doctor?> GetDoctorWithDetailsAsync(int id)
    {
        return await _dbSet
            .Include(d => d.Specialty)
            .Include(d => d.User)
            .Include(d => d.Reviews)
            .FirstOrDefaultAsync(d => d.Id == id);
    }
}
