using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/profile")]
    [Authorize]
    public class ProfileController(IPatientService patientService) : ControllerBase
    {
        // GET /profile/me
        [HttpGet("me")]
        [ProducesResponseType(typeof(PatientDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> GetMyProfile()
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized(new { message = "Invalid token." });

            var patient = await patientService.GetPatientByIdAsync(patientId);
            if (patient is null)
                return NotFound(new { message = "Patient not found." });

            return Ok(new
            {
                id = patient.Id.ToString(),
                name = patient.FullName,
                email = patient.Email,
                phone = patient.PhoneNumber,
                role = "Patient",
                dateOfBirth = patient.DateOfBirth.ToString("yyyy-MM-dd")
            });
        }

        // PUT /profile/me
        [HttpPut("me")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> UpdateMyProfile([FromBody] UpdateProfileDto dto)
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized(new { message = "Invalid token." });

            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            // Map to UpdatePatientDto
            var existing = await patientService.GetPatientByIdAsync(patientId);
            if (existing is null)
                return NotFound(new { message = "Patient not found." });

            var updateDto = new UpdatePatientDto
            {
                Id = patientId,
                FullName = dto.Name ?? existing.FullName,
                PhoneNumber = dto.Phone ?? existing.PhoneNumber,
                Email = existing.Email,
                DateOfBirth = dto.BirthDate ?? dto.DateOfBirth ?? existing.DateOfBirth,
                Gender = existing.Gender,
                IsActive = existing.IsActive
            };

            try
            {
                var updated = await patientService.UpdatePatientAsync(updateDto);
                if (updated is null)
                    return NotFound(new { message = "Patient not found." });

                return Ok(new
                {
                    message = "Profile updated",
                    name = updated.FullName,
                    phone = updated.PhoneNumber,
                    birthDate = updated.DateOfBirth.ToString("yyyy-MM-dd"),
                    dateOfBirth = updated.DateOfBirth.ToString("yyyy-MM-dd")
                });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new { message = ex.Message });
            }
        }
    }

    /// <summary>
    /// Lightweight DTO matching exactly what the frontend sends
    /// </summary>
    public class UpdateProfileDto
    {
        public string? Name { get; set; }
        public string? Phone { get; set; }
        public DateTime? BirthDate { get; set; }
        public DateTime? DateOfBirth { get; set; }
    }
}
