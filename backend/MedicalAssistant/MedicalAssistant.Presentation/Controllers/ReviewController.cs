using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ReviewsController : ControllerBase
    {
        private readonly IReviewService _reviewService;

        public ReviewsController(IReviewService reviewService)
        {
            _reviewService = reviewService;
        }

        // GET: api/reviews/doctor/5
        [HttpGet("doctor/{doctorId}")]
        public async Task<IActionResult> GetReviewsByDoctor(int doctorId)
        {
            var reviews = await _reviewService.GetReviewsByDoctorIdAsync(doctorId);
            return Ok(reviews);
        }

        // GET: api/reviews/5
        [HttpGet("{id}")]
        public async Task<IActionResult> GetReview(int id)
        {
            var review = await _reviewService.GetReviewByIdAsync(id);

            if (review == null)
                return NotFound();

            return Ok(review);
        }

        // GET: api/reviews/recent/5
        [HttpGet("recent/{count}")]
        public async Task<IActionResult> GetRecentReviews(int count)
        {
            var reviews = await _reviewService.GetRecentReviewsAsync(count);
            return Ok(reviews);
        }

        // GET: api/reviews/top/5
        [HttpGet("top/{count}")]
        public async Task<IActionResult> GetTopRatedReviews(int count)
        {
            var reviews = await _reviewService.GetTopRatedReviewsAsync(count);
            return Ok(reviews);
        }

        // GET: api/reviews/doctor/5/rating
        [HttpGet("doctor/{doctorId}/rating")]
        public async Task<IActionResult> GetDoctorAverageRating(int doctorId)
        {
            var rating = await _reviewService.GetDoctorAverageRatingAsync(doctorId);
            return Ok(rating);
        }

        // POST: api/reviews
        [HttpPost]
        public async Task<IActionResult> CreateReview([FromBody] CreateReviewDTO dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var author = User?.Identity?.Name ?? "Anonymous";

            var review = await _reviewService.CreateReviewAsync(dto, author);

            return CreatedAtAction(nameof(GetReview), new { id = review.Id }, review);
        }


    }
}