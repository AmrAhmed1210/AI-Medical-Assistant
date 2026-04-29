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
    [Route("api/patients")]
    [Authorize]
    public class MedicalProfilesController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;

        public MedicalProfilesController(IPatientRecordService patientRecordService)
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

        // POST /api/patients/{id}/profile
        [HttpPost("{id:int}/profile")]
        [Authorize(Roles = "Doctor,Nurse")]
        public async Task<IActionResult> Create(int id, [FromBody] CreateMedicalProfileDto dto)
        {
            if (dto == null) return BadRequest(new { message = "Invalid payload." });

            var profile = new MedicalProfile
            {
                BloodType = dto.BloodType,
                WeightKg = dto.WeightKg,
                HeightCm = dto.HeightCm,
                IsSmoker = dto.IsSmoker,
                SmokingDetails = dto.SmokingDetails,
                DrinksAlcohol = dto.DrinksAlcohol,
                ExerciseHabits = dto.ExerciseHabits,
                EmergencyContactName = dto.EmergencyContactName,
                EmergencyContactPhone = dto.EmergencyContactPhone,
                EmergencyContactRelation = dto.EmergencyContactRelation,
            };

            var created = await _patientRecordService.CreateMedicalProfileAsync(id, profile);
            return CreatedAtAction(nameof(Get), new { id = id }, created);
        }

        // GET /api/patients/{id}/profile
        [HttpGet("{id:int}/profile")]
        public async Task<IActionResult> Get(int id)
        {
            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? string.Empty;
            var currentPatientId = GetPatientIdFromClaims();
            if (!role.Equals("Doctor", StringComparison.OrdinalIgnoreCase) && currentPatientId != id)
                return Forbid();

            var profile = await _patientRecordService.GetMedicalProfileAsync(id);
            if (profile == null) return NotFound(new { message = "Medical profile not found." });
            return Ok(profile);
        }

        // PATCH /api/patients/{id}/profile
        [HttpPatch("{id:int}/profile")]
        [Authorize(Roles = "Doctor,Nurse")]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateMedicalProfileDto dto)
        {
            if (dto == null) return BadRequest(new { message = "Invalid payload." });

            var updates = new MedicalProfile
            {
                BloodType = dto.BloodType,
                WeightKg = dto.WeightKg,
                HeightCm = dto.HeightCm,
                IsSmoker = dto.IsSmoker ?? false,
                SmokingDetails = dto.SmokingDetails,
                DrinksAlcohol = dto.DrinksAlcohol ?? false,
                ExerciseHabits = dto.ExerciseHabits,
                EmergencyContactName = dto.EmergencyContactName,
                EmergencyContactPhone = dto.EmergencyContactPhone,
                EmergencyContactRelation = dto.EmergencyContactRelation,
            };

            var updated = await _patientRecordService.UpdateMedicalProfileAsync(id, updates);
            if (updated == null) return NotFound(new { message = "Medical profile not found." });
            return Ok(updated);
        }
    }
}
