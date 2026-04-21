using MedicalAssistant.Services_Abstraction.Contracts;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/users")]
    [Authorize]
    public class FollowController : ControllerBase
    {
        private readonly IPatientService _patientService;

        public FollowController(IPatientService patientService)
        {
            _patientService = patientService;
        }

        // GET /api/users/followed-doctors
        [HttpGet("followed-doctors")]
        public async Task<IActionResult> GetFollowedDoctors()
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized();

            var dtos = await _patientService.GetFollowedDoctorsAsync(patientId);
            return Ok(dtos);
        }

        // POST /api/users/follow/{doctorId}
        [HttpPost("follow/{doctorId}")]
        public async Task<IActionResult> FollowDoctor(int doctorId)
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized();

            await _patientService.FollowDoctorAsync(patientId, doctorId);
            return Ok(new { message = "Doctor followed successfully" });
        }

        // DELETE /api/users/unfollow/{doctorId}
        [HttpDelete("unfollow/{doctorId}")]
        public async Task<IActionResult> UnfollowDoctor(int doctorId)
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value ?? User.FindFirst("UserId")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized();

            var result = await _patientService.UnfollowDoctorAsync(patientId, doctorId);
            if (!result) return NotFound();

            return Ok(new { message = "Doctor unfollowed successfully" });
        }
    }
}
