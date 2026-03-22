using AutoMapper;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;

namespace MedicalAssistant.Services.Services
{
    public class ReviewService : IReviewService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IMapper _mapper;

        public ReviewService(IUnitOfWork unitOfWork, IMapper mapper)
        {
            _unitOfWork = unitOfWork;
            _mapper = mapper;
        }

        public async Task<IEnumerable<ReviewDto>> GetReviewsByDoctorIdAsync(int doctorId)
        {
            var reviews = await _unitOfWork.Reviews.GetByDoctorIdAsync(doctorId);
            return _mapper.Map<IEnumerable<ReviewDto>>(reviews);
        }

        public async Task<ReviewDto?> GetReviewByIdAsync(int reviewId)
        {
            var review = await _unitOfWork.Reviews.GetByIdAsync(reviewId);
            return review == null ? null : _mapper.Map<ReviewDto>(review);
        }

        public async Task<IEnumerable<ReviewDto>> GetRecentReviewsAsync(int count)
        {
            var reviews = await _unitOfWork.Reviews.GetRecentReviewsAsync(count);
            return _mapper.Map<IEnumerable<ReviewDto>>(reviews);
        }

        public async Task<IEnumerable<ReviewDto>> GetTopRatedReviewsAsync(int count)
        {
            var reviews = await _unitOfWork.Reviews.GetTopRatedReviewsAsync(count);
            return _mapper.Map<IEnumerable<ReviewDto>>(reviews);
        }

        public async Task<double> GetDoctorAverageRatingAsync(int doctorId)
        {
            return await _unitOfWork.Reviews.GetDoctorAverageRatingAsync(doctorId);
        }

        public async Task<int> GetDoctorReviewsCountAsync(int doctorId)
        {
            return await _unitOfWork.Reviews.GetDoctorReviewsCountAsync(doctorId);
        }

        public async Task<(IEnumerable<ReviewDto> Items, int TotalCount)> GetPaginatedByDoctorAsync(int doctorId, int pageNumber, int pageSize)
        {
            var (items, totalCount) = await _unitOfWork.Reviews.GetPaginatedByDoctorAsync(doctorId, pageNumber, pageSize);
            var mappedItems = _mapper.Map<IEnumerable<ReviewDto>>(items);
            return (mappedItems, totalCount);
        }

        public async Task<ReviewDto> CreateReviewAsync(CreateReviewDTO dto, string author)
        {
            // التحقق مما إذا كان المستخدم قد قيم هذا الطبيب مسبقاً
            if (await _unitOfWork.Reviews.HasUserReviewedDoctorAsync(dto.DoctorId, author))
                throw new InvalidOperationException("You have already reviewed this doctor.");

            var review = _mapper.Map<Review>(dto);
            review.Author = author;
            review.CreatedAt = DateTime.UtcNow;

            await _unitOfWork.Reviews.AddAsync(review);
            await _unitOfWork.SaveChangesAsync();

            return _mapper.Map<ReviewDto>(review);
        }

        public async Task<bool> UpdateReviewAsync(int reviewId, UpdateReviewDto dto)
        {
            var review = await _unitOfWork.Reviews.GetByIdAsync(reviewId);
            if (review == null) return false;

            review.Rating = dto.Rating;
            review.Comment = dto.Comment;

            _unitOfWork.Reviews.Update(review);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> DeleteReviewAsync(int reviewId)
        {
            var review = await _unitOfWork.Reviews.GetByIdAsync(reviewId);
            if (review == null) return false;

            _unitOfWork.Reviews.Delete(review);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> HasUserReviewedDoctorAsync(int doctorId, string author)
        {
            return await _unitOfWork.Reviews.HasUserReviewedDoctorAsync(doctorId, author);
        }
    }
}