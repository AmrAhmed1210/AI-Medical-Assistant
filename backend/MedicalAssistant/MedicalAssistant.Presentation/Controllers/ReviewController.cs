using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("reviews")]
    public class ReviewsController : ControllerBase
    {
        private readonly IReviewService _reviewService;

        public ReviewsController(IReviewService reviewService)
        {
            _reviewService = reviewService;
        }

        // GET /reviews/{doctorId}
        [HttpGet("{doctorId}")]
        public async Task<IActionResult> GetReviewsByDoctor(int doctorId)
        {
            var reviews = await _reviewService.GetReviewsByDoctorIdAsync(doctorId);
            return Ok(reviews);
        }

        // POST /reviews
        [HttpPost]
        [Authorize]
        public async Task<IActionResult> CreateReview([FromBody] CreateReviewDTO dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var author = User.FindFirst("name")?.Value
                      ?? User.Identity?.Name
                      ?? "Anonymous";

            var review = await _reviewService.CreateReviewAsync(dto, author);
            return Ok(review);
        }
    }
}