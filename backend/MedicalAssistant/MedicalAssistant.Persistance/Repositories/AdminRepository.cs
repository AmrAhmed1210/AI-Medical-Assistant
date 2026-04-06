using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class AdminRepository : IAdminRepository
    {
        private readonly MedicalAssistantDbContext _context;

        public AdminRepository(MedicalAssistantDbContext context)
        {
            _context = context;
        }

        public async Task<int> GetCountAsync<T>() where T : class
        {
            return await _context.Set<T>().CountAsync();
        }

        public async Task<(IEnumerable<User> Items, int TotalCount)> GetUsersAsync(int page, int pageSize, string? search, string? role)
        {
            var query = _context.Set<User>().Where(u => !u.IsDeleted).AsQueryable();

            if (!string.IsNullOrEmpty(role))
            {
                query = query.Where(u => u.Role == role);
            }

            if (!string.IsNullOrEmpty(search))
            {
                query = query.Where(u => u.FullName.Contains(search) || u.Email.Contains(search));
            }

            var totalCount = await query.CountAsync();
            var items = await query
                .OrderByDescending(u => u.CreatedAt)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            return (items, totalCount);
        }

        public async Task<bool> UpdateUserStatusAsync(int userId, bool isActive)
        {
            var user = await _context.Set<User>().FindAsync(userId);
            if (user == null) return false;

            user.IsActive = isActive;
            return true;
        }

        public async Task<bool> DeleteUserAsync(int userId)
        {
            var user = await _context.Set<User>().FindAsync(userId);
            if (user == null) return false;

            user.IsDeleted = true;
            return true;
        }
    }
}