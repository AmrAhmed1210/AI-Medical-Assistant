using System;
using System.Linq;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api")]
    [Authorize]
    public class SurgeriesController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;

        public SurgeriesController(IPatientRecordService patientRecordService)
        {
            _patientRecordService = patientRecordService;
        }

        private int GetPatientIdFromClaims()
        {
            var claim = User.FindFirst("PatientId")?.Value
                        ?? User.FindFirst("UserId")?.Value
                        ?? User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
            return int.TryParse(claim, out var id) ? id : 0;
        }

        // GET /api/patients/{id}/surgeries
        [HttpGet("patients/{id:int}/surgeries")]
        public async Task<IActionResult> GetForPatient(int id)
        {
            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? string.Empty;
            var currentPatientId = GetPatientIdFromClaims();
            if (!role.Equals("Doctor", StringComparison.OrdinalIgnoreCase) && currentPatientId != id)
                return Forbid();

            var surgeries = await _patientRecordService.GetSurgeriesAsync(id);
            return Ok(surgeries.OrderByDescending(s => s.CreatedAt));
        }

        // POST /api/patients/{id}/surgeries
        [HttpPost("patients/{id:int}/surgeries")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> CreateForPatient(int id, [FromBody] CreateSurgeryDto dto)
        {
            if (dto == null) return BadRequest(new { message = "Invalid payload." });

            var surgery = new SurgeryHistory
            {
                SurgeryName = dto.SurgeryName ?? string.Empty,
                SurgeryDate = dto.SurgeryDate ?? DateOnly.FromDateTime(DateTime.UtcNow),
                HospitalName = dto.HospitalName,
                DoctorName = dto.DoctorName,
                Complications = dto.Complications,
                Notes = dto.Notes,
            };

            var created = await _patientRecordService.AddSurgeryAsync(id, surgery);
            return CreatedAtAction(nameof(GetForPatient), new { id = id }, created);
        }

        // PATCH /api/surgeries/{id}
        [HttpPatch("surgeries/{surgeryId:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Update(int surgeryId, [FromBody] UpdateSurgeryDto dto)
        {
            var updates = new SurgeryHistory
            {
                SurgeryName = dto.SurgeryName,
                SurgeryDate = dto.SurgeryDate ?? default,
                HospitalName = dto.HospitalName,
                DoctorName = dto.DoctorName,
                Complications = dto.Complications,
                Notes = dto.Notes,
            };

            var updated = await _patientRecordService.UpdateSurgeryAsync(surgeryId, updates);
            if (updated == null) return NotFound(new { message = "Surgery not found." });
            return Ok(updated);
        }

        // DELETE /api/surgeries/{id}
        [HttpDelete("surgeries/{surgeryId:int}")]
        [Authorize(Roles = "Doctor,Admin")]
        public async Task<IActionResult> Delete(int surgeryId)
        {
            var deleted = await _patientRecordService.DeleteSurgeryAsync(surgeryId);
            if (!deleted) return NotFound(new { message = "Surgery not found." });
            return Ok(new { message = "Surgery removed." });
        }
    }
}
