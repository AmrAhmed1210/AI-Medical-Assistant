using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/profile")]
    [Authorize]
    public class ProfileController(IPatientService patientService, IPhotoService photoService, IDoctorService doctorService, IUnitOfWork _unitOfWork) : ControllerBase
    {
        [HttpPost("photo")]
        public async Task<IActionResult> UploadPhoto(IFormFile file)
        {
            if (file == null || file.Length == 0) return BadRequest("Photo is required");

            var userIdClaim = User.FindFirst("UserId")?.Value ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(userIdClaim, out var userId))
                return Unauthorized();

            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? "Patient";
            var url = await photoService.UploadPhotoAsync(file);

            if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctor = (await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == userId)).FirstOrDefault();
                if (doctor != null) await doctorService.UpdatePhotoAsync(doctor.Id, url);
            }
            else
            {
                var patient = (await _unitOfWork.Repository<Patient>().FindAsync(p => p.UserId == userId)).FirstOrDefault();
                if (patient != null) await patientService.UpdatePhotoAsync(patient.Id, url);
            }

            return Ok(new { photoUrl = url });
        }

        [HttpPost("upload")]
        public async Task<IActionResult> UploadFile(IFormFile file)
        {
            if (file == null || file.Length == 0) return BadRequest("File is required");
            var url = await photoService.UploadFileAsync(file);
            return Ok(new { url, fileName = file.FileName });
        }

        // GET /profile/me
        [HttpGet("me")]
        [ProducesResponseType(typeof(PatientDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> GetMyProfile()
        {
            var userIdClaim = User.FindFirst("UserId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(userIdClaim, out var userId))
                return Unauthorized(new { message = "Invalid token." });

            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? "Patient";

            if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doctor = await doctorService.GetMyScheduleAsync(userId); // This returns DoctorScheduleDto but we need DoctorDetailDto
                var doctorProfile = await doctorService.GetProfileAsync(userId);
                if (doctorProfile == null) return NotFound(new { message = "Doctor profile not found." });

                return Ok(new
                {
                    id = doctorProfile.Id.ToString(),
                    name = doctorProfile.FullName,
                    email = doctorProfile.Email,
                    role = "Doctor",
                    photoUrl = doctorProfile.PhotoUrl,
                    isAvailable = doctorProfile.IsAvailable
                });
            }

            var patient = (await _unitOfWork.Repository<Patient>().FindAsync(p => p.UserId == userId)).FirstOrDefault();
            if (patient is null)
                return NotFound(new { message = "Patient profile not found." });

            return Ok(new
            {
                id = patient.Id.ToString(),
                name = patient.FullName,
                email = patient.Email,
                phone = patient.PhoneNumber,
                role = "Patient",
                dateOfBirth = patient.DateOfBirth.ToString("yyyy-MM-dd"),
                photoUrl = patient.ImageUrl
            });
        }

        // PUT /profile/me
        [HttpPut("me")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> UpdateMyProfile([FromBody] UpdateProfileDto dto)
        {
            var userIdClaim = User.FindFirst("UserId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(userIdClaim, out var userId))
                return Unauthorized(new { message = "Invalid token." });

            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            // Map to UpdatePatientDto
            var patient = (await _unitOfWork.Repository<Patient>().FindAsync(p => p.UserId == userId)).FirstOrDefault();
            if (patient is null)
                return NotFound(new { message = "Patient profile not found for update." });

            var updateDto = new UpdatePatientDto
            {
                Id = patient.Id,
                FullName = dto.Name ?? patient.FullName,
                PhoneNumber = dto.Phone ?? patient.PhoneNumber,
                Email = patient.Email,
                DateOfBirth = dto.BirthDate ?? dto.DateOfBirth ?? patient.DateOfBirth,
                Gender = patient.Gender,
                IsActive = patient.IsActive
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
