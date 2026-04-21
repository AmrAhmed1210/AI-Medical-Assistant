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

        public async Task<IEnumerable<ReviewDto>> GetReviewsByDoctorIdAsync(int doctorId, int? patientId = null)
        {
            var reviews = await _unitOfWork.Reviews.GetByDoctorIdAsync(doctorId);
            var dtos = _mapper.Map<IEnumerable<ReviewDto>>(reviews).ToList();

            if (patientId.HasValue)
            {
                foreach (var dto in dtos)
                {
                    var reviewObj = reviews.FirstOrDefault(r => r.Id.ToString() == dto.Id);
                    if (reviewObj != null)
                    {
                        dto.IsMine = (reviewObj.PatientId == patientId.Value);
                    }
                }
            }

            return dtos;
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

        public async Task<(IEnumerable<ReviewDto> Items, int TotalCount)> GetPaginatedByDoctorAsync(int doctorId, int pageNumber, int pageSize, int? patientId = null)
        {
            var (items, totalCount) = await _unitOfWork.Reviews.GetPaginatedByDoctorAsync(doctorId, pageNumber, pageSize);
            var dtos = _mapper.Map<IEnumerable<ReviewDto>>(items).ToList();

            if (patientId.HasValue)
            {
                foreach (var dto in dtos)
                {
                    var reviewObj = items.FirstOrDefault(r => r.Id.ToString() == dto.Id);
                    if (reviewObj != null)
                    {
                        dto.IsMine = (reviewObj.PatientId == patientId.Value);
                    }
                }
            }

            return (dtos, totalCount);
        }

        public async Task<ReviewDto> CreateReviewAsync(CreateReviewDTO dto, string author)
        {
            var finalAuthor = !string.IsNullOrWhiteSpace(dto.Author) ? dto.Author : author;
            
            // Robust check using PatientId if available
            bool alreadyReviewed = false;
            if (dto.PatientId.HasValue)
            {
                alreadyReviewed = await _unitOfWork.Reviews.HasUserReviewedDoctorAsync(dto.DoctorId, dto.PatientId.Value);
            }
            else
            {
                alreadyReviewed = await _unitOfWork.Reviews.HasUserReviewedDoctorAsync(dto.DoctorId, finalAuthor);
            }

            if (alreadyReviewed)
                return null;

            var review = _mapper.Map<Review>(dto);
            if (string.IsNullOrWhiteSpace(review.Author)) review.Author = finalAuthor;
            review.CreatedAt = DateTime.UtcNow;

            await _unitOfWork.Reviews.AddAsync(review);
            await _unitOfWork.SaveChangesAsync();

            await RecalculateDoctorRatingAsync(dto.DoctorId);

            var result = _mapper.Map<ReviewDto>(review);
            result.IsMine = true;
            return result;
        }

        public async Task<bool> UpdateReviewAsync(int reviewId, UpdateReviewDto dto)
        {
            var review = await _unitOfWork.Reviews.GetByIdAsync(reviewId);
            if (review == null) return false;

            review.Rating = dto.Rating;
            review.Comment = dto.Comment;

            _unitOfWork.Reviews.Update(review);
            await _unitOfWork.SaveChangesAsync();
            await RecalculateDoctorRatingAsync(review.DoctorId);
            return true;
        }

        public async Task<bool> DeleteReviewAsync(int reviewId)
        {
            var review = await _unitOfWork.Reviews.GetByIdAsync(reviewId);
            if (review == null) return false;

            var doctorId = review.DoctorId;
            _unitOfWork.Reviews.Delete(review);
            await _unitOfWork.SaveChangesAsync();
            await RecalculateDoctorRatingAsync(doctorId);
            return true;
        }

        public async Task<bool> HasUserReviewedDoctorAsync(int doctorId, string author)
        {
            return await _unitOfWork.Reviews.HasUserReviewedDoctorAsync(doctorId, author);
        }

        public async Task<bool> HasUserReviewedDoctorAsync(int doctorId, int patientId)
        {
            return await _unitOfWork.Reviews.HasUserReviewedDoctorAsync(doctorId, patientId);
        }

        public async Task<ReviewDto?> UpdateMyReviewAsync(int doctorId, string author, UpdateReviewDto dto)
        {
            var review = await _unitOfWork.Reviews.GetByDoctorAndAuthorAsync(doctorId, author);
            if (review == null) return null;

            review.Rating = dto.Rating;
            review.Comment = dto.Comment;
            _unitOfWork.Reviews.Update(review);
            await _unitOfWork.SaveChangesAsync();
            await RecalculateDoctorRatingAsync(doctorId);

            var result = _mapper.Map<ReviewDto>(review);
            result.IsMine = true;
            return result;
        }

        public async Task<ReviewDto?> UpdateMyReviewAsync(int doctorId, int patientId, UpdateReviewDto dto)
        {
            // First try by PatientId
            var review = await _unitOfWork.Reviews.GetByDoctorAndPatientIdAsync(doctorId, patientId);
            
            // If not found, try by author names (migration fallback)
            if (review == null)
            {
                // We need names to check, but since we only have patientId here, 
                // we should let the controller call the name-based overload if needed
                // OR we fetch the patient name here.
                var patient = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.PatientModule.Patient>()
                    .GetByIdAsync(patientId);
                
                if (patient != null)
                {
                    review = await _unitOfWork.Reviews.GetByDoctorAndAuthorAsync(doctorId, patient.FullName);
                    
                    // If found by name, "Migrate" it by setting the PatientId for future robust lookup
                    if (review != null)
                    {
                        review.PatientId = patientId;
                    }
                }
            }

            if (review == null) return null;

            review.Rating = dto.Rating;
            review.Comment = dto.Comment;
            _unitOfWork.Reviews.Update(review);
            await _unitOfWork.SaveChangesAsync();
            await RecalculateDoctorRatingAsync(doctorId);

            var result = _mapper.Map<ReviewDto>(review);
            result.IsMine = true;
            return result;
        }

        public async Task<bool> DeleteMyReviewAsync(int doctorId, string author)
        {
            var review = await _unitOfWork.Reviews.GetByDoctorAndAuthorAsync(doctorId, author);
            if (review == null) return false;

            _unitOfWork.Reviews.Delete(review);
            await _unitOfWork.SaveChangesAsync();
            await RecalculateDoctorRatingAsync(doctorId);
            return true;
        }

        public async Task<bool> DeleteMyReviewAsync(int doctorId, int patientId)
        {
            var review = await _unitOfWork.Reviews.GetByDoctorAndPatientIdAsync(doctorId, patientId);
            
            if (review == null)
            {
                var patient = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.PatientModule.Patient>()
                    .GetByIdAsync(patientId);
                
                if (patient != null)
                {
                    review = await _unitOfWork.Reviews.GetByDoctorAndAuthorAsync(doctorId, patient.FullName);
                }
            }

            if (review == null) return false;

            _unitOfWork.Reviews.Delete(review);
            await _unitOfWork.SaveChangesAsync();
            await RecalculateDoctorRatingAsync(doctorId);
            return true;
        }

        private async Task RecalculateDoctorRatingAsync(int doctorId)
        {
            var doctor = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.DoctorsModule.Doctor>()
                .GetByIdAsync(doctorId);

            if (doctor == null) return;

            var reviews = await _unitOfWork.Reviews.GetByDoctorIdAsync(doctorId);
            var reviewList = reviews.ToList();

            doctor.ReviewCount = reviewList.Count;
            doctor.Rating = reviewList.Count > 0 ? reviewList.Average(r => r.Rating) : 0;

            _unitOfWork.Repository<MedicalAssistant.Domain.Entities.DoctorsModule.Doctor>().Update(doctor);
            await _unitOfWork.SaveChangesAsync();
        }
    }
}
