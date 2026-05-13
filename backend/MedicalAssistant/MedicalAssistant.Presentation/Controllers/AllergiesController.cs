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
    public class AllergiesController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;

        public AllergiesController(IPatientRecordService patientRecordService)
        {
            _patientRecordService = patientRecordService;
        }

        private int GetPatientIdFromClaims()
        {
            var claim = User.FindFirst("PatientId")?.Value;
            return int.TryParse(claim, out var id) ? id : 0;
        }

        private bool IsDoctorOrOwner(int patientId)
        {
            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? string.Empty;
            if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase)) return true;
            var currentPatientId = GetPatientIdFromClaims();
            return currentPatientId == patientId;
        }

        // GET /api/patients/{id}/allergies
        [HttpGet("patients/{id:int}/allergies")]
        public async Task<IActionResult> GetForPatient(int id)
        {
            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? string.Empty;
            var currentPatientId = GetPatientIdFromClaims();
            if (!role.Equals("Doctor", StringComparison.OrdinalIgnoreCase) && currentPatientId != id)
                return Forbid();

            var allergies = await _patientRecordService.GetAllergiesAsync(id);
            return Ok(allergies.OrderByDescending(a => a.CreatedAt));
        }

        // POST /api/patients/{id}/allergies
        [HttpPost("patients/{id:int}/allergies")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> CreateForPatient(int id, [FromBody] CreateAllergyDto dto)
        {
            if (!IsDoctorOrOwner(id)) return Forbid();
            if (dto == null) return BadRequest(new { message = "Invalid payload." });

            var allergy = new AllergyRecord
            {
                AllergyType = dto.AllergyType ?? string.Empty,
                AllergenName = dto.AllergenName ?? string.Empty,
                Severity = dto.Severity ?? string.Empty,
                ReactionDescription = dto.ReactionDescription,
                FirstObservedDate = dto.FirstObservedDate,
                IsActive = dto.IsActive ?? true,
            };

            var created = await _patientRecordService.AddAllergyAsync(id, allergy);
            return CreatedAtAction(nameof(GetForPatient), new { id = id }, created);
        }

        // PATCH /api/allergies/{id}
        [HttpPatch("allergies/{allergyId:int}")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> Update(int allergyId, [FromBody] UpdateAllergyDto dto)
        {
            var updates = new AllergyRecord
            {
                AllergyType = dto.AllergyType,
                AllergenName = dto.AllergenName,
                Severity = dto.Severity,
                ReactionDescription = dto.ReactionDescription,
                FirstObservedDate = dto.FirstObservedDate ?? default,
                IsActive = dto.IsActive ?? true,
            };

            var updated = await _patientRecordService.UpdateAllergyAsync(allergyId, updates);
            if (updated == null) return NotFound(new { message = "Allergy not found." });
            return Ok(updated);
        }

        // DELETE /api/allergies/{id}
        [HttpDelete("allergies/{allergyId:int}")]
        [Authorize(Roles = "Doctor,Patient,Admin")]
        public async Task<IActionResult> Delete(int allergyId)
        {
            var deleted = await _patientRecordService.DeleteAllergyAsync(allergyId);
            if (!deleted) return NotFound(new { message = "Allergy not found." });
            return Ok(new { message = "Allergy removed." });
        }
    }
}
