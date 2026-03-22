using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;

namespace MedicalAssistant.Services.Services
{
    /// <summary>
    /// Patient service implementation.
    /// Contains business rules for creating/updating/searching and managing patients.
    /// </summary>
    public class PatientService(IUnitOfWork unitOfWork) : IPatientService
    {
        public async Task<IEnumerable<PatientDto>> GetAllPatientsAsync()
        {
            var patients = await unitOfWork.Patients.GetAllAsync();
            return patients.Select(MapToDto);
        }

        public async Task<PatientDto?> GetPatientByIdAsync(int id)
        {
            var patient = await unitOfWork.Patients.GetByIdAsync(id);
            return patient is null ? null : MapToDto(patient);
        }

        public async Task<PatientDto?> GetPatientByEmailAsync(string email)
        {
            var patient = await unitOfWork.Patients.GetByEmailAsync(email);
            return patient is null ? null : MapToDto(patient);
        }

        public async Task<PatientDto?> GetPatientByPhoneNumberAsync(string phoneNumber)
        {
            var patient = await unitOfWork.Patients.GetByPhoneNumberAsync(phoneNumber);
            return patient is null ? null : MapToDto(patient);
        }

        public async Task<IEnumerable<PatientDto>> GetActivePatientsAsync()
        {
            var patients = await unitOfWork.Patients.GetActivePatients();
            return patients.Select(MapToDto);
        }

        public async Task<IEnumerable<PatientDto>> GetInactivePatientsAsync()
        {
            var patients = await unitOfWork.Patients.GetInactivePatients();
            return patients.Select(MapToDto);
        }

        public async Task<IEnumerable<PatientDto>> SearchPatientsByNameAsync(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return Array.Empty<PatientDto>();

            var patients = await unitOfWork.Patients.SearchByNameAsync(name.Trim());
            return patients.Select(MapToDto);
        }

        public async Task<IEnumerable<PatientDto>> GetPatientsByBloodTypeAsync(string bloodType)
        {
            if (string.IsNullOrWhiteSpace(bloodType))
                return Array.Empty<PatientDto>();

            var patients = await unitOfWork.Patients.GetByBloodTypeAsync(bloodType.Trim());
            return patients.Select(MapToDto);
        }

        public async Task<PaginatedResultDto<PatientDto>> GetPaginatedPatientsAsync(int pageNumber, int pageSize)
        {
            pageNumber = pageNumber < 1 ? 1 : pageNumber;
            pageSize = pageSize < 1 ? 10 : pageSize;

            var (items, totalCount) = await unitOfWork.Patients.GetPaginatedAsync(pageNumber, pageSize);

            return new PaginatedResultDto<PatientDto>
            {
                Items = items.Select(MapToDto),
                TotalCount = totalCount,
                PageNumber = pageNumber,
                PageSize = pageSize
            };
        }

        public async Task<PatientDto> CreatePatientAsync(CreatePatientDto createPatientDto)
        {
            ArgumentNullException.ThrowIfNull(createPatientDto);

            var email = createPatientDto.Email.Trim();
            var phone = createPatientDto.PhoneNumber.Trim();

            if (await unitOfWork.Patients.EmailExistsAsync(email))
                throw new InvalidOperationException("Email already exists.");

            if (await unitOfWork.Patients.PhoneNumberExistsAsync(phone))
                throw new InvalidOperationException("Phone number already exists.");

            var patient = new Patient
            {
                FullName = createPatientDto.FullName.Trim(),
                Email = email,
                PhoneNumber = phone,
                DateOfBirth = createPatientDto.DateOfBirth,
                Gender = createPatientDto.Gender.Trim(),
                Address = createPatientDto.Address?.Trim(),
                ImageUrl = createPatientDto.ImageUrl?.Trim(),
                BloodType = createPatientDto.BloodType?.Trim(),
                MedicalNotes = createPatientDto.MedicalNotes?.Trim(),

                CreatedAt = DateTime.UtcNow,
                IsActive = true
            };

            await unitOfWork.Patients.AddAsync(patient);
            await unitOfWork.SaveChangesAsync();

            return MapToDto(patient);
        }

        public async Task<PatientDto?> UpdatePatientAsync(UpdatePatientDto updatePatientDto)
        {
            ArgumentNullException.ThrowIfNull(updatePatientDto);

            var patient = await unitOfWork.Patients.GetByIdAsync(updatePatientDto.Id);
            if (patient is null) return null;

            var email = updatePatientDto.Email.Trim();
            var phone = updatePatientDto.PhoneNumber.Trim();

            var existingByEmail = await unitOfWork.Patients.GetByEmailAsync(email);
            if (existingByEmail is not null && existingByEmail.Id != updatePatientDto.Id)
                throw new InvalidOperationException("Email already exists for another patient.");

            var existingByPhone = await unitOfWork.Patients.GetByPhoneNumberAsync(phone);
            if (existingByPhone is not null && existingByPhone.Id != updatePatientDto.Id)
                throw new InvalidOperationException("Phone number already exists for another patient.");

            patient.FullName = updatePatientDto.FullName.Trim();
            patient.Email = email;
            patient.PhoneNumber = phone;
            patient.DateOfBirth = updatePatientDto.DateOfBirth;
            patient.Gender = updatePatientDto.Gender.Trim();
            patient.Address = updatePatientDto.Address?.Trim();
            patient.ImageUrl = updatePatientDto.ImageUrl?.Trim();
            patient.BloodType = updatePatientDto.BloodType?.Trim();
            patient.MedicalNotes = updatePatientDto.MedicalNotes?.Trim();
            patient.IsActive = updatePatientDto.IsActive;

            unitOfWork.Patients.Update(patient);
            await unitOfWork.SaveChangesAsync();

            return MapToDto(patient);
        }

        public async Task<bool> DeletePatientAsync(int id)
        {
            var patient = await unitOfWork.Patients.GetByIdAsync(id);
            if (patient is null) return false;

            unitOfWork.Patients.Delete(patient);
            await unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> ActivatePatientAsync(int id)
        {
            var patient = await unitOfWork.Patients.GetByIdAsync(id);
            if (patient is null) return false;

            if (patient.IsActive) return true;

            patient.IsActive = true;
            unitOfWork.Patients.Update(patient);
            await unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> DeactivatePatientAsync(int id)
        {
            var patient = await unitOfWork.Patients.GetByIdAsync(id);
            if (patient is null) return false;

            if (!patient.IsActive) return true;

            patient.IsActive = false;
            unitOfWork.Patients.Update(patient);
            await unitOfWork.SaveChangesAsync();
            return true;
        }

        public Task<bool> EmailExistsAsync(string email)
        {
            if (string.IsNullOrWhiteSpace(email)) return Task.FromResult(false);
            return unitOfWork.Patients.EmailExistsAsync(email.Trim());
        }

        public Task<bool> PhoneNumberExistsAsync(string phoneNumber)
        {
            if (string.IsNullOrWhiteSpace(phoneNumber)) return Task.FromResult(false);
            return unitOfWork.Patients.PhoneNumberExistsAsync(phoneNumber.Trim());
        }

        private static PatientDto MapToDto(Patient patient)
        {
            return new PatientDto
            {
                Id = patient.Id,
                FullName = patient.FullName,
                Email = patient.Email,
                PhoneNumber = patient.PhoneNumber,
                DateOfBirth = patient.DateOfBirth,
                Gender = patient.Gender,
                Address = patient.Address,
                ImageUrl = patient.ImageUrl,
                BloodType = patient.BloodType,
                MedicalNotes = patient.MedicalNotes,
                CreatedAt = patient.CreatedAt,
                IsActive = patient.IsActive
            };
        }
    }
}
