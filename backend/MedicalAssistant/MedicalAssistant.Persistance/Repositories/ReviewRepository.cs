using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Repositories
{
    public class ReviewRepository : GenericRepository<Review>, IReviewRepository
    {
        private new readonly MedicalAssistantDbContext _context;

        public ReviewRepository(MedicalAssistantDbContext context) : base(context)
        {
            _context = context;
        }

        public async Task<IEnumerable<Review>> GetByDoctorIdAsync(int doctorId)
        {
            return await _context.Reviews
                .Where(r => r.DoctorId == doctorId)
                .Include(r => r.Doctor)
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }

        public async Task<IEnumerable<Review>> GetRecentReviewsAsync(int count)
        {
            return await _context.Reviews
                .Include(r => r.Doctor)
                .OrderByDescending(r => r.CreatedAt)
                .Take(count)
                .ToListAsync();
        }

        public async Task<IEnumerable<Review>> GetTopRatedReviewsAsync(int count)
        {
            return await _context.Reviews
                .Include(r => r.Doctor)
                .OrderByDescending(r => r.Rating)
                .Take(count)
                .ToListAsync();
        }

        public async Task<double> GetDoctorAverageRatingAsync(int doctorId)
        {
            return await _context.Reviews
                .Where(r => r.DoctorId == doctorId)
                .AverageAsync(r => (double?)r.Rating) ?? 0;
        }

        public async Task<int> GetDoctorReviewsCountAsync(int doctorId)
        {
            return await _context.Reviews
                .Where(r => r.DoctorId == doctorId)
                .CountAsync();
        }

        public async Task<bool> HasUserReviewedDoctorAsync(int doctorId, string author)
        {
            var normalizedAuthor = (author ?? string.Empty).Trim().ToLower();
            if (string.IsNullOrWhiteSpace(normalizedAuthor))
            {
                return false;
            }

            return await _context.Reviews
                .AnyAsync(r =>
                    r.DoctorId == doctorId &&
                    (
                        ((r.Author ?? string.Empty).Trim().ToLower() == normalizedAuthor) ||
                        ((r.PatientName ?? string.Empty).Trim().ToLower() == normalizedAuthor)
                    ));
        }

        public async Task<bool> HasUserReviewedDoctorAsync(int doctorId, int patientId)
        {
            return await _context.Reviews
                .AnyAsync(r => r.DoctorId == doctorId && r.PatientId == patientId);
        }

        public async Task<Review?> GetByDoctorAndAuthorAsync(int doctorId, string author)
        {
            var normalizedAuthor = (author ?? string.Empty).Trim().ToLower();
            if (string.IsNullOrWhiteSpace(normalizedAuthor))
            {
                return null;
            }

            return await _context.Reviews
                .Where(r =>
                    r.DoctorId == doctorId &&
                    (
                        ((r.Author ?? string.Empty).Trim().ToLower() == normalizedAuthor) ||
                        ((r.PatientName ?? string.Empty).Trim().ToLower() == normalizedAuthor)
                    ))
                .OrderByDescending(r => r.CreatedAt)
                .FirstOrDefaultAsync();
        }

        public async Task<Review?> GetByDoctorAndPatientIdAsync(int doctorId, int patientId)
        {
            return await _context.Reviews
                .Where(r => r.DoctorId == doctorId && r.PatientId == patientId)
                .OrderByDescending(r => r.CreatedAt)
                .FirstOrDefaultAsync();
        }

        public async Task<(IEnumerable<Review> Items, int TotalCount)> GetPaginatedByDoctorAsync(
            int doctorId,
            int pageNumber,
            int pageSize)
        {
            var query = _context.Reviews
                .Where(r => r.DoctorId == doctorId)
                .Include(r => r.Doctor);

            var totalCount = await query.CountAsync();

            var items = await query
                .OrderByDescending(r => r.CreatedAt)
                .Skip((pageNumber - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();

            return (items, totalCount);
        }

        public async Task<IEnumerable<Review>> GetReviewsWithDoctorAsync()
        {
            return await _context.Reviews
                .Include(r => r.Doctor)
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }

        public async Task<IEnumerable<Review>> SearchReviewsAsync(string keyword)
        {
            return await _context.Reviews
                .Include(r => r.Doctor)
                .Where(r => r.Comment.Contains(keyword))
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }
    }
}
