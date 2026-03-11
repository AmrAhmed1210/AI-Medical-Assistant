using MedicalAssistant.Domain.Entities;
using MedicalAssistant.Domain.Entities.ReviewsModule;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IReviewRepository : IGenericRepository<Review>
    {
        Task<IEnumerable<Review>> GetByDoctorIdAsync(int doctorId);

        Task<IEnumerable<Review>> GetRecentReviewsAsync(int count);

        Task<IEnumerable<Review>> GetTopRatedReviewsAsync(int count);

        Task<double> GetDoctorAverageRatingAsync(int doctorId);

        Task<int> GetDoctorReviewsCountAsync(int doctorId);

        Task<bool> HasUserReviewedDoctorAsync(int doctorId, string author);

        Task<(IEnumerable<Review> Items, int TotalCount)> GetPaginatedByDoctorAsync(
            int doctorId,
            int pageNumber,
            int pageSize);

        Task<IEnumerable<Review>> GetReviewsWithDoctorAsync();

        Task<IEnumerable<Review>> SearchReviewsAsync(string keyword);
    }
}