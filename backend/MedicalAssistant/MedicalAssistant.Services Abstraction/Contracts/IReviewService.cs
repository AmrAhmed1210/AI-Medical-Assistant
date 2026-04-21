using MedicalAssistant.Shared.DTOs.ReviewDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IReviewService
    {
        Task<IEnumerable<ReviewDto>> GetReviewsByDoctorIdAsync(int doctorId, int? patientId = null);

        Task<ReviewDto?> GetReviewByIdAsync(int reviewId);

        Task<IEnumerable<ReviewDto>> GetRecentReviewsAsync(int count);

        Task<IEnumerable<ReviewDto>> GetTopRatedReviewsAsync(int count);

        Task<double> GetDoctorAverageRatingAsync(int doctorId);

        Task<int> GetDoctorReviewsCountAsync(int doctorId);

        Task<(IEnumerable<ReviewDto> Items, int TotalCount)> GetPaginatedByDoctorAsync(
            int doctorId,
            int pageNumber,
            int pageSize,
            int? patientId = null);

        Task<ReviewDto> CreateReviewAsync(CreateReviewDTO dto, string author);

        Task<bool> UpdateReviewAsync(int reviewId, UpdateReviewDto dto);

        Task<bool> DeleteReviewAsync(int reviewId);

        Task<bool> HasUserReviewedDoctorAsync(int doctorId, string author);
        Task<bool> HasUserReviewedDoctorAsync(int doctorId, int patientId);

        Task<ReviewDto?> UpdateMyReviewAsync(int doctorId, string author, UpdateReviewDto dto);
        Task<ReviewDto?> UpdateMyReviewAsync(int doctorId, int patientId, UpdateReviewDto dto);

        Task<bool> DeleteMyReviewAsync(int doctorId, string author);
        Task<bool> DeleteMyReviewAsync(int doctorId, int patientId);
    }
}
