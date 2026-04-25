using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/reviews")]
    public class ReviewsController : ControllerBase
    {
        private readonly IReviewService _reviewService;
        private readonly IUnitOfWork _unitOfWork;

        public ReviewsController(IReviewService reviewService, IUnitOfWork unitOfWork)
        {
            _reviewService = reviewService;
            _unitOfWork = unitOfWork;
        }

        // GET /reviews/{doctorId}
        [HttpGet("{doctorId}")]
        public async Task<IActionResult> GetReviewsByDoctor(int doctorId)
        {
            int? patientId = null;
            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (int.TryParse(patientIdClaim, out var pid)) patientId = pid;

            var reviews = await _reviewService.GetReviewsByDoctorIdAsync(doctorId, patientId);
            return Ok(reviews);
        }

        // POST /reviews
        [HttpPost]
        [Authorize]
        public async Task<IActionResult> CreateReview([FromBody] CreateReviewDTO dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var aliases = BuildIdentityAliases();
            dto.Author = aliases.FirstOrDefault() ?? "Anonymous";
            dto.PatientName ??= aliases.FirstOrDefault();

            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (int.TryParse(patientIdClaim, out var pid)) dto.PatientId = pid;

            var review = await _reviewService.CreateReviewAsync(dto, dto.Author);
            if (review == null)
                return Conflict(new { message = "You have already reviewed this doctor." });
            return Ok(review);
        }

        // PUT /api/reviews/{doctorId}/mine
        [HttpPut("{doctorId}/mine")]
        [Authorize]
        public async Task<IActionResult> UpdateMyReview(int doctorId, [FromBody] UpdateReviewDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (int.TryParse(patientIdClaim, out var patientId))
            {
                var updated = await _reviewService.UpdateMyReviewAsync(doctorId, patientId, dto);
                if (updated != null) return Ok(updated);
            }

            foreach (var alias in BuildIdentityAliases())
            {
                var updated = await _reviewService.UpdateMyReviewAsync(doctorId, alias, dto);
                if (updated != null)
                    return Ok(updated);
            }

            return NotFound(new { message = "You don't have a review for this doctor yet." });
        }

        // DELETE /api/reviews/{doctorId}/mine
        [HttpDelete("{doctorId}/mine")]
        [Authorize]
        public async Task<IActionResult> DeleteMyReview(int doctorId)
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (int.TryParse(patientIdClaim, out var patientId))
            {
                var deleted = await _reviewService.DeleteMyReviewAsync(doctorId, patientId);
                if (deleted) return Ok(new { message = "Review deleted successfully." });
            }

            foreach (var alias in BuildIdentityAliases())
            {
                var deleted = await _reviewService.DeleteMyReviewAsync(doctorId, alias);
                if (deleted)
                    return Ok(new { message = "Review deleted successfully." });
            }

            return NotFound(new { message = "You don't have a review for this doctor yet." });
        }

        // DELETE /api/reviews/{reviewId} - Delete by review ID (for mobile)
        [HttpDelete("{reviewId}")]
        [Authorize]
        public async Task<IActionResult> DeleteReview(int reviewId)
        {
            if (!await IsCurrentPatientReviewOwner(reviewId))
                return Forbid("You can only delete your own reviews.");

            var deleted = await _reviewService.DeleteReviewAsync(reviewId);
            if (deleted) return Ok(new { message = "Review deleted successfully." });

            return NotFound(new { message = "Failed to delete review." });
        }

        // PUT /api/reviews/{reviewId} - Update by review ID (for mobile)
        [HttpPut("{reviewId:int}")]
        [Authorize]
        public async Task<IActionResult> UpdateReview(int reviewId, [FromBody] UpdateReviewDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            if (!await IsCurrentPatientReviewOwner(reviewId))
                return Forbid("You can only update your own reviews.");

            var updated = await _reviewService.UpdateReviewAsync(reviewId, dto);
            if (!updated) return NotFound(new { message = "Review not found." });

            var review = await _reviewService.GetReviewByIdAsync(reviewId);
            return review == null ? Ok(new { message = "Review updated successfully." }) : Ok(review);
        }

        private List<string> BuildIdentityAliases()
        {
            var aliases = new List<string>
            {
                User.FindFirst("name")?.Value ?? string.Empty,
                User.Identity?.Name ?? string.Empty,
                User.FindFirst(ClaimTypes.Email)?.Value ?? string.Empty,
            };

            return aliases
                .Select(alias => alias.Trim())
                .Where(alias => !string.IsNullOrWhiteSpace(alias))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        private int? GetCurrentPatientId()
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            return int.TryParse(patientIdClaim, out var patientId) ? patientId : null;
        }

        private async Task<bool> IsCurrentPatientReviewOwner(int reviewId)
        {
            var review = await _unitOfWork.Repository<Review>().GetByIdAsync(reviewId);
            if (review == null) return false;

            var patientId = GetCurrentPatientId();
            if (patientId.HasValue && review.PatientId.HasValue && review.PatientId.Value == patientId.Value)
            {
                return true;
            }

            var aliases = BuildIdentityAliases();
            if (aliases.Count == 0) return false;

            return aliases.Any(alias =>
                string.Equals(alias, review.Author, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(alias, review.PatientName, StringComparison.OrdinalIgnoreCase));
        }
    }
}
