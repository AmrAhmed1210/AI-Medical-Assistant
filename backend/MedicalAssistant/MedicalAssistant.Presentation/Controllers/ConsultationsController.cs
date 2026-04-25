using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.ConsultationDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/consultations")]
    [Authorize]
    public class ConsultationsController : ControllerBase
    {
        private readonly IConsultationService _consultationService;
        private readonly IUnitOfWork _unitOfWork;

        public ConsultationsController(IConsultationService consultationService, IUnitOfWork unitOfWork)
        {
            _consultationService = consultationService;
            _unitOfWork = unitOfWork;
        }

        // POST /api/consultations
        [HttpPost]
        [Authorize(Roles = "Doctor")]
        [ProducesResponseType(typeof(ConsultationDto), StatusCodes.Status201Created)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> Create([FromBody] CreateConsultationDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            // Get doctorId from JWT token
            var doctorIdClaim = User.FindFirst("DoctorId")?.Value;
            if (!int.TryParse(doctorIdClaim, out var doctorId))
            {
                // Try to get via UserId if DoctorId not directly available
                var userIdClaim = User.FindFirst("UserId")?.Value
                               ?? User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                if (!int.TryParse(userIdClaim, out var userId))
                    return Unauthorized(new { message = "Invalid token." });

                var doctor = (await _unitOfWork.Doctors.FindAsync(d => d.UserId == userId))
                    .FirstOrDefault();
                if (doctor == null)
                    return Unauthorized(new { message = "Doctor profile not found." });

                doctorId = doctor.Id;
            }

            var consultation = await _consultationService.CreateConsultationAsync(doctorId, dto);
            return CreatedAtAction(nameof(GetById), new { id = consultation.Id }, consultation);
        }

        // GET /api/consultations/my
        [HttpGet("my")]
        [Authorize(Roles = "Patient")]
        [ProducesResponseType(typeof(IEnumerable<ConsultationDto>), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetMyConsultations()
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized(new { message = "Invalid token." });

            var consultations = await _consultationService.GetConsultationsByPatientIdAsync(patientId);
            return Ok(consultations);
        }

        // GET /api/consultations/doctor
        [HttpGet("doctor")]
        [Authorize(Roles = "Doctor")]
        [ProducesResponseType(typeof(IEnumerable<ConsultationDto>), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetDoctorConsultations()
        {
            var doctorIdClaim = User.FindFirst("DoctorId")?.Value;
            if (!int.TryParse(doctorIdClaim, out var doctorId))
            {
                var userIdClaim = User.FindFirst("UserId")?.Value
                               ?? User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                if (!int.TryParse(userIdClaim, out var userId))
                    return Unauthorized(new { message = "Invalid token." });

                var doctor = (await _unitOfWork.Doctors.FindAsync(d => d.UserId == userId))
                    .FirstOrDefault();
                if (doctor == null)
                    return Unauthorized(new { message = "Doctor profile not found." });

                doctorId = doctor.Id;
            }

            var consultations = await _consultationService.GetConsultationsByDoctorIdAsync(doctorId);
            return Ok(consultations);
        }

        // GET /api/consultations/{id}
        [HttpGet("{id}")]
        [ProducesResponseType(typeof(ConsultationDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> GetById(int id)
        {
            var consultation = await _consultationService.GetConsultationByIdAsync(id);
            if (consultation == null)
                return NotFound(new { message = "Consultation not found." });

            if (!await CanAccessConsultationAsync(consultation))
                return Forbid();

            return Ok(consultation);
        }

        // PUT /api/consultations/{id}
        [HttpPut("{id}")]
        [ProducesResponseType(typeof(ConsultationDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateConsultationDto dto)
        {
            dto.Id = id;

            var current = await _consultationService.GetConsultationByIdAsync(id);
            if (current == null) return NotFound(new { message = "Consultation not found." });
            if (!await CanAccessConsultationAsync(current)) return Forbid();

            var updated = await _consultationService.UpdateConsultationAsync(dto);
            if (updated == null) return NotFound(new { message = "Consultation not found." });

            return Ok(updated);
        }

        // PUT /api/consultations/{id}/complete
        [HttpPut("{id}/complete")]
        [Authorize(Roles = "Doctor")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Complete(int id)
        {
            var current = await _consultationService.GetConsultationByIdAsync(id);
            if (current == null) return NotFound(new { message = "Consultation not found." });
            if (!await CanDoctorAccessConsultationAsync(current)) return Forbid();

            var success = await _consultationService.CompleteConsultationAsync(id);
            if (!success) return NotFound(new { message = "Consultation not found." });

            return Ok(new { message = "Consultation marked as completed." });
        }

        // PUT /api/consultations/{id}/cancel
        [HttpPut("{id}/cancel")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Cancel(int id)
        {
            var current = await _consultationService.GetConsultationByIdAsync(id);
            if (current == null) return NotFound(new { message = "Consultation not found." });
            if (!await CanAccessConsultationAsync(current)) return Forbid();

            var success = await _consultationService.CancelConsultationAsync(id);
            if (!success) return NotFound(new { message = "Consultation not found." });

            return Ok(new { message = "Consultation cancelled." });
        }

        // DELETE /api/consultations/{id}
        [HttpDelete("{id}")]
        [Authorize(Roles = "Doctor")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Delete(int id)
        {
            var current = await _consultationService.GetConsultationByIdAsync(id);
            if (current == null) return NotFound(new { message = "Consultation not found." });
            if (!await CanDoctorAccessConsultationAsync(current)) return Forbid();

            var success = await _consultationService.DeleteConsultationAsync(id);
            if (!success) return NotFound(new { message = "Consultation not found." });

            return Ok(new { message = "Consultation deleted." });
        }

        private async Task<bool> CanDoctorAccessConsultationAsync(ConsultationDto consultation)
        {
            var doctorIdClaim = User.FindFirst("DoctorId")?.Value;
            if (int.TryParse(doctorIdClaim, out var doctorId) && doctorId > 0)
                return consultation.DoctorId == doctorId;

            var userIdClaim = User.FindFirst("UserId")?.Value
                           ?? User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (!int.TryParse(userIdClaim, out var userId))
                return false;

            var doctor = (await _unitOfWork.Doctors.FindAsync(d => d.UserId == userId))
                .FirstOrDefault();

            return doctor != null && consultation.DoctorId == doctor.Id;
        }

        private async Task<bool> CanAccessConsultationAsync(ConsultationDto consultation)
        {
            var role = User.FindFirst(ClaimTypes.Role)?.Value
                    ?? User.FindFirst("role")?.Value
                    ?? string.Empty;

            if (string.Equals(role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                return await CanDoctorAccessConsultationAsync(consultation);
            }

            if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
            {
                var patientIdClaim = User.FindFirst("PatientId")?.Value
                                  ?? User.FindFirst("sub")?.Value;
                if (int.TryParse(patientIdClaim, out var patientId))
                    return consultation.PatientId == patientId;
            }

            return false;
        }
    }
}
